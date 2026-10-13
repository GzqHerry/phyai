"""Protocol-v1 messages: one UTF-8 JSON frame followed by raw numeric frames.

JSON describes shape, dtype (including byte order), field names and operation
metadata. Each following frame is one array in C order. Both peers validate the
protocol version and byte counts before reconstructing detached arrays, so they
need no matching Python object layout, NumPy ABI, Torch or ROS installation.

Observations use one array per sensor field. Action chunks use one stacked array
per action field with the prediction horizon as its leading dimension. Robot
receive timestamps stay in the robot's monotonic clock domain throughout RPC.
"""

from __future__ import annotations

import json
import math
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Mapping

import numpy as np

from ..types import Sample, ActionChunk, FeatureSpec, Observation
from ..validation import validate_schema, validate_values

PROTOCOL_VERSION = 1
MAX_BYTES = 128 * 1024 * 1024


def _schema(schema):
    """Normalize a physical schema for exact JSON comparison on either peer."""
    validate_schema(schema)
    return {
        key: dict(
            shape=list(s.shape),
            dtype=np.dtype(s.dtype).str,
            unit=s.unit,
            names=None if s.names is None else list(s.names),
            frame=s.frame,
        )
        for key, s in schema.items()
    }


def _read_schema(value):
    """Rebuild field specifications; PolicySpec.from_dict validates them next."""
    return {
        k: FeatureSpec(
            tuple(v["shape"]),
            v["dtype"],
            v["unit"],
            None if v["names"] is None else tuple(v["names"]),
            v["frame"],
        )
        for k, v in value.items()
    }


@dataclass(frozen=True)
class PolicySpec:
    """Physical data contract plus model metadata needed before opening hardware.

    Schemas include units and coordinate order, not just tensor dimensions.
    Metadata is application-owned (for example horizon and image size); it is
    advertised by the server but is not part of the physical-schema equality
    check. The entry point validates metadata needed to configure its robot.
    """

    embodiment: str
    observation_schema: Mapping[str, FeatureSpec]
    action_schema: Mapping[str, FeatureSpec]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return dict(
            embodiment=self.embodiment,
            observation_schema=_schema(self.observation_schema),
            action_schema=_schema(self.action_schema),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_dict(cls, value):
        spec = cls(
            value["embodiment"],
            _read_schema(value["observation_schema"]),
            _read_schema(value["action_schema"]),
            value["metadata"],
        )
        spec.to_dict()
        return spec

    def check(self, embodiment, observation_schema, action_schema):
        """Reject another embodiment or any physical field-contract mismatch."""
        if embodiment != self.embodiment:
            raise ValueError(f"embodiment mismatch: expected {self.embodiment}")
        for label, actual, expected in (
            ("observation", observation_schema, self.observation_schema),
            ("action", action_schema, self.action_schema),
        ):
            if _schema(actual) != _schema(expected):
                raise ValueError(f"{label} schema mismatch")


def pack(meta, arrays=()):
    """Copy numeric arrays into a bounded multipart message.

    Strided inputs retain their values and shape, including zero-dimensional
    arrays. Object/string/complex arrays and non-finite JSON metadata are not
    part of this protocol. Array finiteness is checked by the schema validators.
    """
    frames, descriptors = [], []
    for value in arrays:
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
            raise ValueError("wire values must be real numeric numpy arrays")
        # tobytes(order='C') handles strides without changing scalar shape.
        descriptors.append(dict(shape=list(value.shape), dtype=value.dtype.str))
        frames.append(value.tobytes())
    header = json.dumps(
        dict(meta, protocol=PROTOCOL_VERSION, arrays=descriptors),
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(header) + sum(map(len, frames)) > MAX_BYTES:
        raise ValueError("policy message exceeds size limit")
    return [header, *frames]


def unpack(frames):
    """Validate framing and return metadata plus independent NumPy arrays.

    The shape/product check precedes reconstruction. Copies detach values from
    transport buffers so later socket operations cannot alter an observation.
    This validates the wire format; field/schema validation follows separately.
    """
    if not frames or sum(map(len, frames)) > MAX_BYTES:
        raise ValueError("invalid policy message size")
    header = json.loads(frames[0])
    if not isinstance(header, dict):
        raise ValueError("policy metadata must be a JSON object")
    if (
        type(header.get("protocol")) is not int
        or header["protocol"] != PROTOCOL_VERSION
    ):
        raise ValueError("policy protocol version mismatch")
    descriptors = header.pop("arrays")
    if len(descriptors) != len(frames) - 1:
        raise ValueError("array frame count mismatch")
    arrays = []
    for desc, data in zip(descriptors, frames[1:]):
        dtype, shape = np.dtype(desc["dtype"]), tuple(desc["shape"])
        if dtype.kind not in "biuf" or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid array dtype or shape")
        if math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError("array frame size mismatch")
        arrays.append(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
    return header, arrays


def observation_payload(observation):
    """Keep each field name and robot receive timestamp aligned with its frame."""
    return dict(
        samples=[
            dict(key=k, received_at_ns=s.received_at_ns)
            for k, s in observation.samples.items()
        ]
    ), [s.value for s in observation.samples.values()]


def read_observation(header, arrays, schema):
    """Reject duplicate/missing fields and reconstruct the original robot snapshot."""
    descriptors = header["samples"]
    if len(descriptors) != len(arrays) or len({d["key"] for d in descriptors}) != len(
        arrays
    ):
        raise ValueError("observation field count mismatch")
    samples = {}
    for d, value in zip(descriptors, arrays):
        stamp = d["received_at_ns"]
        # These timestamps belong to the robot. Never compare to the server clock.
        if type(stamp) is not int or stamp < 0:
            raise ValueError("invalid robot receive timestamp")
        samples[d["key"]] = Sample(value, stamp)
    validate_values({k: s.value for k, s in samples.items()}, schema)
    return Observation(samples)


def chunk_payload(chunk, schema):
    """Validate physical targets and stack each field over the full horizon."""
    if not isinstance(chunk, ActionChunk) or not chunk.actions:
        raise ValueError("expected nonempty ActionChunk")
    if not math.isfinite(chunk.step_period_s) or chunk.step_period_s <= 0:
        raise ValueError("invalid action period")
    for action in chunk.actions:
        validate_values(action, schema)
    keys = list(schema)
    return dict(keys=keys, step_period_s=chunk.step_period_s), [
        np.stack([a[k] for a in chunk.actions]) for k in keys
    ]


def read_chunk(header, arrays, schema):
    """Restore per-step actions; leave execution-prefix selection to deployment."""
    keys = header["keys"]
    if (
        len(keys) != len(set(keys))
        or set(keys) != set(schema)
        or len(keys) != len(arrays)
    ):
        raise ValueError("action field mismatch")
    count = len(arrays[0]) if arrays else 0
    if not count or any(len(a) != count for a in arrays):
        raise ValueError("action horizon mismatch")
    chunk = ActionChunk(
        tuple({k: a[i].copy() for k, a in zip(keys, arrays)} for i in range(count)),
        header["step_period_s"],
    )
    chunk_payload(chunk, schema)
    return chunk
