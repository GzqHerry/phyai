"""Model-size RGB images for camera decoders; requires the 'images' extra.

Decoders call resize_with_pad on their receive worker, so cached samples are
already small and never copy native-resolution frames into a snapshot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resize_with_pad(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Resize RGB with PIL bilinear filtering into a black-padded square.

    The longest side becomes size and the other keeps its aspect ratio after
    integer truncation. This matches the PIL backend of PhyAI's pi0.5
    processor, so a server receiving an already-sized image leaves it as is.
    Input and output are HWC uint8 arrays; the output owns its storage.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise ImportError(
            "resize_with_pad requires Pillow; install phyai-robot[images]"
        ) from error

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB uint8 HWC image")
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    height, width = image.shape[:2]
    if (height, width) == (size, size):
        return np.array(image)
    ratio = max(width / size, height / size)
    resized = Image.fromarray(image).resize(
        (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.array(padded)
