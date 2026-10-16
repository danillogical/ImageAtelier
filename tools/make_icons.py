"""Generate the PWA icons into webstudio/static/icons/ (run: .venv/bin/python tools/make_icons.py)."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "webstudio" / "static" / "icons"
TOP, BOTTOM = (124, 140, 255), (176, 108, 255)


def gradient(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size - 2)
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
    return img


def sparkle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill) -> None:
    """Four-point star."""
    k = r * 0.22
    draw.polygon(
        [(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k), (cx, cy + r), (cx - k, cy + k), (cx - r, cy), (cx - k, cy - k)],
        fill=fill,
    )


def glyph(size: int, scale: float) -> Image.Image:
    """Transparent layer with a large sparkle and a small one; scale shrinks it toward the centre."""
    big = size * 4
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = big / 2
    sparkle(d, c - big * 0.04 * scale, c + big * 0.04 * scale, big * 0.30 * scale, (255, 255, 255, 255))
    sparkle(d, c + big * 0.20 * scale, c - big * 0.20 * scale, big * 0.11 * scale, (255, 255, 255, 235))
    return layer.resize((size, size), Image.LANCZOS)


def icon(size: int, *, rounded: bool, scale: float) -> Image.Image:
    img = gradient(size).convert("RGBA")
    img.alpha_composite(glyph(size, scale))
    if rounded:
        mask = Image.new("L", (size * 4, size * 4), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, size * 4 - 1, size * 4 - 1], radius=size * 4 * 0.225, fill=255)
        img.putalpha(mask.resize((size, size), Image.LANCZOS))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    icon(192, rounded=True, scale=1).save(OUT / "icon-192.png")
    icon(512, rounded=True, scale=1).save(OUT / "icon-512.png")
    # Maskable: full bleed, glyph inside the central 80% safe zone.
    icon(512, rounded=False, scale=0.8).save(OUT / "maskable-512.png")
    # iOS applies its own corner mask and dislikes transparency.
    icon(180, rounded=False, scale=1).convert("RGB").save(OUT / "apple-touch-icon.png")
    icon(32, rounded=True, scale=1).save(OUT / "favicon-32.png")


if __name__ == "__main__":
    main()
