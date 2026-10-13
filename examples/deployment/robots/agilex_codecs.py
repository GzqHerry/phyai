"""AgileX joint/gripper layout bindings for standard ROS messages."""

import numpy as np
from phyai_robot.backends.ros2.codecs import decode_joint_positions, decode_rgb_image

from ..layouts.agilex import JOINT_NAMES

# Kept as an import alias for callers using the deployment image decoder.
decode_image = decode_rgb_image


def decode_joint_state(message):
    """Select joint0..joint6, preserving driver radians and gripper meters."""
    return decode_joint_positions(message, JOINT_NAMES)


def decode_state_fields(message, side):
    """Split one seven-value feedback into six arm joints and one gripper."""
    values = decode_joint_state(message)
    return {
        f"joint_position_{side}": values[:6].copy(),
        f"gripper_{side}": values[6:].copy(),
    }


def encode_state_fields(values, side, message_type):
    """Join the physical fields without clipping, scaling or changing joint order.

    The backend supplies the ROS timestamp after encoding. Keeping the message
    class injected lets offline tests exercise this mapping without importing ROS.
    """
    message = message_type()
    message.name = list(JOINT_NAMES)
    message.position = np.concatenate(
        [values[f"joint_position_{side}"], values[f"gripper_{side}"]]
    ).tolist()
    return message
