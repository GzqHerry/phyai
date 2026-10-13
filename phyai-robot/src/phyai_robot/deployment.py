"""One inference worker and one independently paced, single-step control loop.

The worker takes a fresh snapshot and predicts whenever the scheduler asks for
a new chunk. The control loop sends the scheduler's command for each tick at
control_hz while the scheduler is active, and sends nothing while it is idle.
The default StopAndRefill scheduler pauses motion during every inference; see
scheduling.py for how a real-time chunking strategy fits the same loop.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any, Generic, TypeVar
from _thread import LockType
from dataclasses import dataclass

from .robot import Robot, wait_for_observation
from .types import ActionChunk
from .policy import Policy, PolicyAdapter
from ._compat import BaseExceptionGroup
from .scheduling import Scheduler
from .validation import freeze_values, validate_values


@dataclass(frozen=True)
class DeploymentOptions:
    control_hz: float = 200.0  # Robot.send_action rate while commands are scheduled.
    max_sample_age_s: float = 0.5  # Maximum local receive age of every snapshot field.
    observation_timeout_s: float = 5.0  # Per-snapshot wait for complete, fresh sensors.
    shutdown_timeout_s: float = 2.0  # Maximum wait for an in-flight prediction.
    max_control_lateness_s: float = 0.01  # Allowed delay past a scheduled send.

    def __post_init__(self) -> None:
        for name in (
            "control_hz",
            "max_sample_age_s",
            "observation_timeout_s",
            "shutdown_timeout_s",
            "max_control_lateness_s",
        ):
            value: Any = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if 1e9 / self.control_hz < 1:
            raise ValueError("control period must be at least one nanosecond")


RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class RobotDeployment(Generic[RequestT, ResultT]):
    """Run one observation/inference worker and one independent control loop.

    A Condition protects the scheduler and worker failures. Robot is not
    required to be thread-safe: an I/O lock serializes worker reads, control
    writes, shutdown and external stop requests. predict() never holds that
    lock, so stopping hardware does not wait for a stalled model.
    """

    def __init__(
        self,
        *,
        robot: Robot,
        policy: Policy[RequestT, ResultT],
        adapter: PolicyAdapter[RequestT, ResultT],
        scheduler: Scheduler,
        options: DeploymentOptions,
    ) -> None:
        scheduler.bind(robot.action_schema)
        self.robot = robot
        self.policy = policy
        self.adapter = adapter
        self.scheduler = scheduler
        self.options = options
        self._condition = threading.Condition()
        self._io_lock: LockType = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_error: BaseException | None = None
        self._started = False

    def stop(self) -> None:
        """Request shutdown; no new Robot I/O starts after this method returns.

        The run thread performs Robot.stop/close. Calls already inside bounded
        Robot I/O finish first. In-flight inference is discarded on return.
        """
        # Publish cancellation before waiting for the I/O barrier, so a worker
        # finishing read() cannot start another send while stop() waits its turn.
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        with self._io_lock:
            pass  # Wait for any already-started Robot call to finish.

    def prepare_chunk(self, chunk: ActionChunk) -> ActionChunk:
        """Validate a whole prediction and detach it from the adapter's buffers.

        A malformed target outside the executed prefix still indicates a broken
        prediction, so every target is checked before the scheduler sees it.
        """
        if not isinstance(chunk, ActionChunk) or not chunk.actions:
            raise ValueError("adapter must return a nonempty ActionChunk")
        for action in chunk.actions:
            validate_values(action, self.robot.action_schema)
        prepared = ActionChunk(tuple(freeze_values(a) for a in chunk.actions))
        with self._condition:
            self.scheduler.check(prepared)
        return prepared

    def _produce(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_event.is_set()
                        or self.scheduler.wants_inference()
                    )
                    if self._stop_event.is_set():
                        return
                # Never relax the age threshold: retry until every field is fresh.
                observation = wait_for_observation(
                    self.robot,
                    timeout_s=self.options.observation_timeout_s,
                    max_age_s=self.options.max_sample_age_s,
                    lock=self._io_lock,
                    stop_event=self._stop_event,
                )
                if observation is None:
                    return
                with self._condition:
                    context = self.scheduler.begin_inference(time.monotonic_ns())
                request = self.adapter.to_request(observation, context)
                if self._stop_event.is_set():
                    return
                result = self.policy.predict(request)
                if self._stop_event.is_set():
                    return
                chunk = self.prepare_chunk(self.adapter.to_actions(result, observation))
                with self._condition:
                    if self._stop_event.is_set():
                        return
                    self.scheduler.add_chunk(chunk, context, time.monotonic_ns())
                    self._condition.notify_all()
        except BaseException as error:  # noqa: BLE001 - forward failures across threads
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()
            self.stop()

    def _raise_worker_error(self) -> None:
        with self._condition:
            if self._worker_error is not None:
                raise self._worker_error

    def _execute(self, remaining_steps: int | None) -> int:
        """Send scheduled commands on a fresh clock until the scheduler idles.

        Tolerated lateness skips old ticks instead of replaying them in a
        catch-up burst; larger lateness stops execution. Returns the number of
        commands actually sent.
        """
        period_ns = 1e9 / self.options.control_hz
        max_lateness_ns = self.options.max_control_lateness_s * 1e9
        started_at_ns = time.monotonic_ns()
        tick = 0
        sent = 0
        previous_send_ns = 0
        while True:
            scheduled_tick = tick
            deadline_ns = started_at_ns + round(tick * period_ns)
            if self._stop_event.wait(max(0, (deadline_ns - time.monotonic_ns()) / 1e9)):
                return sent
            woke_at_ns = time.monotonic_ns()
            with self._io_lock:
                if self._stop_event.is_set():
                    return sent
                locked_at_ns = time.monotonic_ns()
                # Keep the original deadline for the lateness check. Moving it
                # forward first would hide stalls larger than the configured limit.
                # Within that limit, select the latest due tick.
                tick = max(tick, math.floor((locked_at_ns - started_at_ns) / period_ns))
                with self._condition:
                    action = self.scheduler.action_at(
                        started_at_ns + round(tick * period_ns)
                    )
                    # The worker waits on this condition; wake it as soon as
                    # the scheduler wants a prediction, idle or not.
                    if action is None or self.scheduler.wants_inference():
                        self._condition.notify_all()
                    if action is None:
                        return sent
                now_ns = time.monotonic_ns()
                lateness_ns = now_ns - deadline_ns
                if lateness_ns > max_lateness_ns:
                    raise TimeoutError(
                        "control deadline exceeded: "
                        f"lateness_ms={lateness_ns / 1e6:.6f}, "
                        f"limit_ms={max_lateness_ns / 1e6:.6f}, "
                        f"scheduled_tick={scheduled_tick}, "
                        f"control_hz={self.options.control_hz:g}, "
                        f"wake_lateness_ms={(woke_at_ns - deadline_ns) / 1e6:.6f}, "
                        f"io_wait_ms={(locked_at_ns - woke_at_ns) / 1e6:.6f}, "
                        f"prepare_ms={(now_ns - locked_at_ns) / 1e6:.6f}, "
                        f"previous_send_ms={previous_send_ns / 1e6:.6f}"
                    )
                self.robot.send_action(action)
                previous_send_ns = time.monotonic_ns() - now_ns
            sent += 1
            if remaining_steps is not None and sent >= remaining_steps:
                return sent
            # Preparation can also cross a tick boundary. Do not replay that tick
            # immediately after this send; keep the next deadline on this clock.
            tick = max(tick + 1, math.floor((now_ns - started_at_ns) / period_ns) + 1)

    def run(self, *, max_steps: int | None = None) -> None:
        """Connect, execute, and always stop/close; max_steps counts actual sends."""
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        if self._started:
            raise RuntimeError("deployment instances are single-use")
        self._started = True
        worker: threading.Thread | None = None
        errors: list[BaseException] = []
        try:
            with self._io_lock:
                if not self._stop_event.is_set():
                    self.robot.connect()
            if not self._stop_event.is_set():
                worker = threading.Thread(
                    target=self._produce, name="phyai-policy", daemon=True
                )
                worker.start()
            sent = 0
            while not self._stop_event.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_event.is_set() or self.scheduler.is_active()
                    )
                self._raise_worker_error()
                if self._stop_event.is_set():
                    break
                sent += self._execute(None if max_steps is None else max_steps - sent)
                if max_steps is not None and sent >= max_steps:
                    break
            self._raise_worker_error()
        except BaseException as error:  # noqa: BLE001 - preserve errors through cleanup
            errors.append(error)
        finally:
            self.stop()
            # Stop hardware before waiting for potentially blocked prediction.
            # The I/O lock also excludes the producer's bounded observation read.
            with self._io_lock:
                for cleanup in (self.robot.stop, self.robot.close):
                    try:
                        cleanup()
                    except BaseException as error:  # noqa: BLE001 - preserve cleanup failures
                        errors.append(error)
            if worker is not None:
                worker.join(self.options.shutdown_timeout_s)
                if worker.is_alive():
                    errors.append(
                        TimeoutError(
                            "policy is still running after robot shutdown; "
                            "do not destroy its resources until predict() returns"
                        )
                    )
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("deployment and/or cleanup failed", errors)
