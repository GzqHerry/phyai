"""Tianji physical schemas and lossless geometry seam; no ROS imports."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping

import numpy as np
from phyai_robot import FeatureSpec
from numpy.typing import NDArray

CAMERA_HEIGHT = 1984

CAMERA_WIDTH = 2560

ARM_DOF = 7

GRIPPER_FEEDBACK_SIZE = 5

INPUT_IDLE = 0

INPUT_USER = 3

MODE_JOINT_POSITION = 1

MODE_CARTESIAN_IMPEDANCE = 2

MODE_JOINT_IMPEDANCE = 3

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, ARM_DOF + 1))

EEF_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")

WRENCH_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")

GRIPPER_FEEDBACK_NAMES = (
    "position",
    "velocity",
    "torque",
    "mos_temperature",
    "motor_temperature",
)

OBSERVATION_SCHEMA: Mapping[str, FeatureSpec] = {
    "head_camera": FeatureSpec((CAMERA_HEIGHT, CAMERA_WIDTH, 3), "uint8"),
    "left_wrist_camera": FeatureSpec((CAMERA_HEIGHT, CAMERA_WIDTH, 3), "uint8"),
    "right_wrist_camera": FeatureSpec((CAMERA_HEIGHT, CAMERA_WIDTH, 3), "uint8"),
    "joint_position_left": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "joint_position_right": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "eef_left": FeatureSpec((7,), "float64", names=EEF_NAMES),
    "eef_right": FeatureSpec((7,), "float64", names=EEF_NAMES),
    "wrench_left": FeatureSpec((6,), "float64", names=WRENCH_NAMES),
    "wrench_right": FeatureSpec((6,), "float64", names=WRENCH_NAMES),
    "gripper_feedback_left": FeatureSpec(
        (GRIPPER_FEEDBACK_SIZE,),
        "float32",
        names=GRIPPER_FEEDBACK_NAMES,
    ),
    "gripper_feedback_right": FeatureSpec(
        (GRIPPER_FEEDBACK_SIZE,),
        "float32",
        names=GRIPPER_FEEDBACK_NAMES,
    ),
}

ACTION_SCHEMA: Mapping[str, FeatureSpec] = {
    "joint_position_left": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "joint_position_right": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "gripper_left": FeatureSpec(
        (1,), "float32", unit="normalized", names=("target_position",)
    ),
    "gripper_right": FeatureSpec(
        (1,), "float32", unit="normalized", names=("target_position",)
    ),
}


def resize_rgb(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Resize RGB with PIL bilinear interpolation and centered black padding.

    The decoder caches small images so observation snapshots do not copy native
    camera frames. This only changes image geometry; model normalization and
    tokenization remain in the adapter. The same operation is used for any
    non-resized image passed directly to the adapter.

    The longest side becomes size; the shorter side keeps its aspect ratio after
    integer rounding. Unused pixels are zero (black), not stretched image data.
    Input/output are HWC uint8 RGB arrays. The returned array owns its storage.
    """
    from PIL import Image

    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    resized = Image.fromarray(image).resize(
        (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.array(padded)


def observation_schema(image_size=224):
    return {
        k: replace(v, shape=(image_size, image_size, 3)) if k.endswith("_camera") else v
        for k, v in OBSERVATION_SCHEMA.items()
    }
