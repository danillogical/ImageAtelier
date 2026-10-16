"""Output sizes and cover-fit cropping for uploaded images.

dimensions() replicates mflux's QwenImage21LatentCreator.dimensions exactly (importing the fork pulls in MLX, which is
slow); tests/scratch compare the two. An image saved at exactly dimensions(res, ratio) keeps that size through later
edits, because the edit path recomputes the size from the last reference image's ratio.
"""

from __future__ import annotations

import math

# Model-card aspect ratios (label -> width/height).
PRESETS: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 2400 / 1792,
    "3:4": 1792 / 2400,
    "3:2": 2528 / 1696,
    "2:3": 1696 / 2528,
    "16:9": 2752 / 1536,
    "9:16": 1536 / 2752,
}
MULTIPLE = 32


def dimensions(resolution: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(resolution * resolution * ratio)
    return max(MULTIPLE, round(width / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(width / ratio / MULTIPLE) * MULTIPLE)


def presets_for(resolution: int) -> list[dict]:
    out = []
    for label, ratio in PRESETS.items():
        w, h = dimensions(resolution, ratio)
        out.append({"label": label, "ratio": ratio, "width": w, "height": h})
    return out


def closest(w: int, h: int) -> str:
    """The preset whose ratio is nearest to w:h (smallest |log ratio difference|)."""
    r = math.log(w / h)
    return min(PRESETS, key=lambda k: abs(r - math.log(PRESETS[k])))


def cover_crop_box(src_w: int, src_h: int, ratio: float) -> tuple[int, int, int, int]:
    """The maximal centred box (x, y, w, h) of the given ratio inside the source: scale-to-fill, crop the overflow."""
    if src_w / src_h > ratio:
        h, w = src_h, min(src_w, max(1, round(src_h * ratio)))
    else:
        w, h = src_w, min(src_h, max(1, round(src_w / ratio)))
    return (src_w - w) // 2, (src_h - h) // 2, w, h


def fit_box_to_ratio(box: tuple[float, float, float, float], src_w: int, src_h: int, ratio: float) -> tuple[int, int, int, int]:
    """Clamp a client crop to the source and shrink it (around its centre) to exactly the preset ratio."""
    x, y, w, h = box
    x0, y0 = max(0.0, min(x, src_w - 1)), max(0.0, min(y, src_h - 1))
    x1, y1 = max(x0 + 1, min(x + w, src_w)), max(y0 + 1, min(y + h, src_h))
    w, h = x1 - x0, y1 - y0
    if w / h > ratio:
        nw = h * ratio
        x0 += (w - nw) / 2
        w = nw
    else:
        nh = w / ratio
        y0 += (h - nh) / 2
        h = nh
    iw, ih = max(1, min(src_w, round(w))), max(1, min(src_h, round(h)))
    ix, iy = max(0, min(src_w - iw, round(x0))), max(0, min(src_h - ih, round(y0)))
    return ix, iy, iw, ih
