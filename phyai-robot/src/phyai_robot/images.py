"""Model-size RGB images for camera decoders; requires the 'images' extra.

Decoders call these on their receive worker, so cached samples are already
small and never copy native-resolution frames into a snapshot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _pil():
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise ImportError(
            "image resizing requires Pillow; install phyai-robot[images]"
        ) from error
    return Image


def _check_rgb(image: NDArray[np.uint8]) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB uint8 HWC image")


def resize(image: NDArray[np.uint8], height: int, width: int) -> NDArray[np.uint8]:
    """Resize RGB to exactly height x width with PIL bilinear filtering.

    The aspect ratio is not preserved. Use it to reproduce a recording that
    stretched camera images to a fixed tile size. The output owns its storage.
    """
    _check_rgb(image)
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("height and width must be positive integers")
    if image.shape[:2] == (height, width):
        return np.array(image)
    Image = _pil()
    resized = Image.fromarray(np.ascontiguousarray(image)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized).copy()


def resize_with_pad(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Resize RGB with PIL bilinear filtering into a black-padded square.

    The longest side becomes size and the other keeps its aspect ratio after
    integer truncation. This matches the PIL backend of PhyAI's pi0.5
    processor, so a server receiving an already-sized image leaves it as is.
    Input and output are HWC uint8 arrays; the output owns its storage.
    """
    _check_rgb(image)
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    height, width = image.shape[:2]
    if (height, width) == (size, size):
        return np.array(image)
    Image = _pil()
    ratio = max(width / size, height / size)
    resized = Image.fromarray(np.ascontiguousarray(image)).resize(
        (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.array(padded)
