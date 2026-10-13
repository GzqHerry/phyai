"""Pure sensor_msgs-compatible decoders; importing them never imports ROS."""

import numpy as np


def decode_joint_positions(message, joint_names):
    """Return detached float64 positions in the requested JointState name order.

    joint_names is a reusable sequence of required names. Extra message joints
    are ignored; duplicate names, missing required names, or inconsistent name
    and position counts are errors. This function does not change physical units.
    """
    positions = np.asarray(message.position, dtype=np.float64)
    names = list(message.name)
    if len(names) != len(set(names)) or any(name not in names for name in joint_names):
        raise ValueError(
            f"Expected unique joint names containing {tuple(joint_names)}, got {names}"
        )
    if len(positions) != len(names):
        raise ValueError("JointState names and positions differ in length")
    return np.array(
        [positions[names.index(name)] for name in joint_names], dtype=np.float64
    )


def decode_rgb_image(message):
    """Return contiguous HWC uint8 RGB, respecting ROS Image row padding.

    RGB/BGR/RGBA/BGRA/mono 8-bit encodings are accepted. Alpha is discarded and
    grayscale is repeated across three channels; no resize or normalization is
    applied. The output owns its data rather than aliasing the incoming message.
    NV12 uses a different codec and is deliberately not guessed from buffer size.
    """
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    encoding = str(message.encoding).lower()
    if encoding not in channels:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    count = channels[encoding]
    height, width, stride = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or stride < width * count:
        raise ValueError("Invalid image dimensions or row stride")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * stride:
        raise ValueError("Image buffer length does not match height * step")
    pixels = raw.reshape(height, stride)[:, : width * count].reshape(
        height, width, count
    )
    if count == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif encoding.startswith("bgr"):
        pixels = pixels[:, :, [2, 1, 0]]
    else:
        pixels = pixels[:, :, :3]
    return np.ascontiguousarray(pixels).copy()
