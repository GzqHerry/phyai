"""A robot-level adapter between named robot fields and a vector model contract.

Most VLA checkpoints consume ordered camera images plus one state vector and
predict a matrix of action vectors. VectorAdapter declares, per robot, which
fields fill those vectors and in which order, so model servers never need to
know an embodiment's field names, units or calibration.

Model inputs are named arrays: "image.<camera>" for each RGB image and "state"
for the state vector. Model outputs contain "actions" with shape (H, D).
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .types import Action, Schema, ActionChunk, FeatureSpec, Observation
from .scheduling import InferenceContext
from .validation import validate_schema

IMAGE_PREFIX = "image."
STATE_KEY = "state"
ACTIONS_KEY = "actions"


@dataclass(frozen=True)
class StateSlot:
    """Append components of one observation field to the model state vector.

    limits=(low, high) maps raw values linearly so low -> 0 and high -> 1,
    without clipping, e.g. a gripper position calibrated to its travel range.
    """

    field: str
    index: tuple[int, ...] | None = None  # Components of a 1-D field; None = all.
    limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class CommandSlot:
    """Fill one complete action field from the next model action components.

    The width is the field's size. clip=(low, high) clamps model values before
    casting to the field's dtype; it is a validity range, not a safety limit.
    """

    field: str
    clip: tuple[float, float] | None = None


def _check_range(name: str, value: tuple[float, float] | None) -> None:
    if value is None:
        return
    if len(value) != 2 or not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain two finite values")
    if value[0] >= value[1]:
        raise ValueError(f"{name} must be increasing")


class VectorAdapter:
    """PolicyAdapter from a robot's schemas to named model input/output arrays.

    Images are passed through without copying, so camera fields must already
    use the model's RGB uint8 image size. State is assembled in float64 and
    sent as float32. Every predicted row becomes one complete robot Action.
    """

    def __init__(
        self,
        *,
        observation_schema: Schema,
        action_schema: Schema,
        cameras: Mapping[str, str],  # Model camera name -> observation field.
        state: Sequence[StateSlot],
        commands: Sequence[CommandSlot],
    ) -> None:
        validate_schema(observation_schema)
        validate_schema(action_schema)
        if not cameras:
            raise ValueError("at least one camera is required")
        for name, field in cameras.items():
            spec = observation_schema.get(field)
            if spec is None:
                raise ValueError(f"camera {name!r}: unknown observation {field!r}")
            if spec.dtype != "uint8" or len(spec.shape) != 3 or spec.shape[2] != 3:
                raise ValueError(f"camera {name!r}: {field!r} is not RGB uint8 HWC")
        self._cameras = dict(cameras)
        self._state: list[tuple[str, NDArray[np.intp], tuple[float, float] | None]] = []
        for slot in state:
            spec = observation_schema.get(slot.field)
            if spec is None or len(spec.shape) != 1:
                raise ValueError(
                    f"state field {slot.field!r} must be a 1-D observation"
                )
            index = np.arange(spec.shape[0]) if slot.index is None else slot.index
            if not len(index) or any(not 0 <= i < spec.shape[0] for i in index):
                raise ValueError(f"state field {slot.field!r}: invalid index")
            _check_range(f"{slot.field} limits", slot.limits)
            self._state.append((slot.field, np.asarray(index, np.intp), slot.limits))
        if not self._state:
            raise ValueError("at least one state slot is required")
        self._commands: list[tuple[str, slice, FeatureSpec, Any]] = []
        start = 0
        for slot in commands:
            spec = action_schema.get(slot.field)
            if spec is None or len(spec.shape) != 1:
                raise ValueError(f"command field {slot.field!r} must be a 1-D action")
            _check_range(f"{slot.field} clip", slot.clip)
            width = spec.shape[0]
            self._commands.append(
                (slot.field, slice(start, start + width), spec, slot.clip)
            )
            start += width
        if sorted(field for field, *_ in self._commands) != sorted(action_schema):
            raise ValueError("command slots must cover every action field once")
        self.action_dim = start
        self.state_dim = sum(len(index) for _, index, _ in self._state)
        self.request_schema: Schema = MappingProxyType(
            {
                **{
                    IMAGE_PREFIX + name: observation_schema[field]
                    for name, field in cameras.items()
                },
                STATE_KEY: FeatureSpec((self.state_dim,), "float32"),
            }
        )

    def state_vector(self, observation: Observation) -> NDArray[np.float32]:
        parts = []
        for field, index, limits in self._state:
            values = observation.samples[field].value[index].astype(np.float64)
            if limits is not None:
                low, high = limits
                values = (values - low) / (high - low)
            parts.append(values)
        state = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(state).all():
            raise ValueError("model state contains a non-finite value")
        return state

    def to_request(
        self, observation: Observation, context: InferenceContext
    ) -> Mapping[str, NDArray[Any]]:
        return {
            **{
                IMAGE_PREFIX + name: observation.samples[field].value
                for name, field in self._cameras.items()
            },
            STATE_KEY: self.state_vector(observation),
        }

    def to_actions(
        self, result: Mapping[str, NDArray[Any]], observation: Observation
    ) -> ActionChunk:
        values = np.asarray(result[ACTIONS_KEY])
        if (
            values.ndim != 2
            or values.shape[1] != self.action_dim
            or values.dtype.kind != "f"
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"expected finite float actions with shape (H, {self.action_dim}), "
                f"got {values.dtype} {values.shape}"
            )
        actions: list[Action] = []
        for row in values:
            action = {}
            for field, part, spec, clip in self._commands:
                value = row[part]
                if clip is not None:
                    value = np.clip(value, *clip)
                action[field] = np.asarray(value, dtype=spec.dtype)
            actions.append(action)
        return ActionChunk(tuple(actions))
