"""AgileX physical state/action layout, independent of ROS and Pi05."""

from __future__ import annotations

import numpy as np
from phyai_robot import FeatureSpec

VIEWS = ("front", "left", "right")
SIDES = ("left", "right")
JOINT_NAMES = tuple(f"joint{i}" for i in range(7))

ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec((6,), "float64", unit="rad")
        for side in SIDES
    },
    **{f"gripper_{side}": FeatureSpec((1,), "float64", unit="m") for side in SIDES},
}


def state_from_observation(observation):
    """Build the model's float32 state: left joints/gripper, then right.

    Physical observations stay float64. This conversion is the existing model
    input convention, not a cast applied to backend feedback or control commands.
    """
    return np.concatenate(
        [
            observation.samples[key].value
            for side in SIDES
            for key in (f"joint_position_{side}", f"gripper_{side}")
        ]
    ).astype(np.float32)


def action_from_target(target):
    """Split a 14-value physical target into detached float64 schema fields."""
    target = np.asarray(target, dtype=np.float64).reshape(14)
    return {
        key: target[part].copy()
        for index, side in enumerate(SIDES)
        for key, part in (
            (f"joint_position_{side}", slice(index * 7, index * 7 + 6)),
            (f"gripper_{side}", slice(index * 7 + 6, index * 7 + 7)),
        )
    }


def target_from_action(action):
    """Join complete physical fields in left-arm/gripper, right-arm/gripper order."""
    return np.concatenate(
        [
            action[key]
            for side in SIDES
            for key in (f"joint_position_{side}", f"gripper_{side}")
        ]
    )


def observation_schema(height=480, width=640):
    """Declare raw RGB dimensions plus physical feedback; do not resize pixels."""
    return {
        **{f"{v}_camera": FeatureSpec((height, width, 3), "uint8") for v in VIEWS},
        **ACTION_SCHEMA,
    }
