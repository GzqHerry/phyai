"""Action-chunk scheduling: when to predict and which command each tick sends.

RobotDeployment owns threads, robot I/O and the control clock. A Scheduler owns
only the action timeline. Every method is called while the deployment holds its
condition lock, so implementations need no synchronization of their own and
must return quickly.

StopAndRefill is the default strategy. Real-time chunking (RTC) fits the same
protocol: request inference while a chunk is still executing, report the
targets that will run during inference through InferenceContext.pending, and
merge the new chunk in add_chunk without ever becoming inactive.
"""

from __future__ import annotations

import math
from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .types import Action, ActionChunk, ActionSchema


@dataclass(frozen=True)
class InferenceContext:
    """The scheduler's state when an inference snapshot is taken.

    pending holds the source targets that will still be executed after the
    snapshot, in order. An RTC adapter can pass them to a guided model. The
    default stop-and-refill strategy never executes during inference, so its
    context is always empty.
    """

    pending: tuple[Action, ...] = ()


class Scheduler(Protocol):
    """Own the action timeline between predictions and control ticks."""

    def bind(self, action_schema: ActionSchema) -> None:
        """Validate the strategy settings against the robot's command fields."""
        ...

    def check(self, chunk: ActionChunk) -> None:
        """Reject a validated chunk that this strategy cannot execute."""
        ...

    def wants_inference(self) -> bool:
        """Return whether the worker should take a snapshot and predict now."""
        ...

    def begin_inference(self, now_ns: int) -> InferenceContext:
        """Record that a snapshot was taken at now_ns and describe pending work."""
        ...

    def add_chunk(
        self, chunk: ActionChunk, context: InferenceContext, now_ns: int
    ) -> None:
        """Accept the prediction made for context; now_ns is its arrival time."""
        ...

    def is_active(self) -> bool:
        """Return whether action_at may produce commands without new input."""
        ...

    def action_at(self, tick_ns: int) -> Action | None:
        """Return the command scheduled for tick_ns, or None to stop sending.

        Returning None ends the active period. The deployment restarts its
        control clock when the scheduler becomes active again.
        """
        ...


def interpolate(left: Action, right: Action, fraction: float, keys: frozenset[str]):
    """Linear interpolation for continuous absolute targets; hold other fields.

    Convex interpolation never overshoots either endpoint, unlike an arbitrary
    spline. This is joint/normalized-gripper interpolation, not an SE(3) pose
    interpolator or a velocity limiter. The result arrays are read-only.
    """
    if fraction == 0 or left is right:
        return left
    values: dict[str, Any] = dict(left)
    for key in keys:
        # Work in float64 to avoid overflow in lower-precision schemas, then
        # restore the schema's dtype.
        value = (
            (1 - fraction) * left[key].astype(np.float64)
            + fraction * right[key].astype(np.float64)
        ).astype(left[key].dtype)
        value.flags.writeable = False
        values[key] = value
    return values


class StopAndRefill:
    """Execute one prefix completely, then pause sending until the next chunk.

    Target i is due i / action_hz seconds after the chunk's first control tick;
    the chunk covers [0, N / action_hz), so the final target is held for one
    source interval. The worker predicts only after a chunk fully drains, so
    observation and inference never overlap motion and no command is sent
    between chunks. This is not RTC or a hard real-time scheduler.
    """

    def __init__(
        self,
        *,
        action_hz: float,  # Rate at which source targets are consumed.
        execution_horizon: int | None = None,  # Prefix length; None keeps all.
        interpolate_keys: Iterable[str] | None = None,
        # Continuous fields; None selects every floating-point command field.
    ) -> None:
        if (
            isinstance(action_hz, bool)
            or not math.isfinite(action_hz)
            or action_hz <= 0
        ):
            raise ValueError("action_hz must be finite and positive")
        if execution_horizon is not None and (
            type(execution_horizon) is not int or execution_horizon < 1
        ):
            raise ValueError("execution_horizon must be a positive integer or None")
        self.action_hz = float(action_hz)
        self.execution_horizon = execution_horizon
        self._requested_keys = (
            None if interpolate_keys is None else frozenset(interpolate_keys)
        )
        self._keys: frozenset[str] = frozenset()
        self._chunk: tuple[Action, ...] | None = None
        self._start_ns: int | None = None

    def bind(self, action_schema: ActionSchema) -> None:
        floating = {
            key
            for key, spec in action_schema.items()
            if np.dtype(spec.dtype).kind == "f"
        }
        if self._requested_keys is None:
            self._keys = frozenset(floating)
            return
        if not self._requested_keys <= set(action_schema):
            raise ValueError("interpolate_keys contains unknown command fields")
        if not self._requested_keys <= floating:
            raise ValueError("interpolation requires floating-point command fields")
        self._keys = self._requested_keys

    def check(self, chunk: ActionChunk) -> None:
        if self.execution_horizon is not None and (
            len(chunk.actions) < self.execution_horizon
        ):
            raise ValueError(
                f"chunk has {len(chunk.actions)} targets; "
                f"execution_horizon is {self.execution_horizon}"
            )

    def wants_inference(self) -> bool:
        return self._chunk is None

    def begin_inference(self, now_ns: int) -> InferenceContext:
        return InferenceContext()

    def add_chunk(
        self, chunk: ActionChunk, context: InferenceContext, now_ns: int
    ) -> None:
        self.check(chunk)
        self._chunk = chunk.actions[: self.execution_horizon]
        self._start_ns = None  # The first control tick starts this chunk's clock.

    def is_active(self) -> bool:
        return self._chunk is not None

    def action_at(self, tick_ns: int) -> Action | None:
        if self._chunk is None:
            return None
        if self._start_ns is None:
            self._start_ns = tick_ns
        position = (tick_ns - self._start_ns) * self.action_hz / 1e9
        if position >= len(self._chunk):
            # The final interval has elapsed. Becoming inactive lets the worker
            # observe; nothing is sent until the next chunk arrives.
            self._chunk = self._start_ns = None
            return None
        index = math.floor(position)
        left = self._chunk[index]
        right = self._chunk[min(index + 1, len(self._chunk) - 1)]
        return interpolate(left, right, position - index, self._keys)
