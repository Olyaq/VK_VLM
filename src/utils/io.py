from __future__ import annotations

import os
from typing import Any
from PIL import Image


def coerce_image(x):
    """Convert different image representations to PIL RGB image.

    Supports:
    - str path
    - PIL.Image.Image
    - dict with keys like {'path': ..., 'bytes': ...} (HF datasets Image(decode=False))
    """
    from PIL import Image
    from io import BytesIO

    # PIL image
    if hasattr(x, "convert"):
        try:
            return x.convert("RGB")
        except Exception:
            pass

    # Path
    if isinstance(x, str):
        return Image.open(x).convert("RGB")

    # HF datasets Image(decode=False) returns dict: {'path': ..., 'bytes': ...}
    if isinstance(x, dict):
        path = x.get("path")
        if path:
            return Image.open(path).convert("RGB")
        b = x.get("bytes")
        if b is not None:
            if isinstance(b, memoryview):
                b = b.tobytes()
            return Image.open(BytesIO(b)).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(x)}")
