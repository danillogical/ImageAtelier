"""Generate the PWA / app icons into imageatelier/static/icons/ (run: .venv/bin/python tools/make_icons.py).

Art: an ink-blue sigil (two rings, ticks, runes) behind a crooked witch's hat on parchment, with a gold spark.
All geometry is in unit coordinates (0..1) and rendered at 4x, then downscaled with LANCZOS.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "imageatelier" / "static" / "icons"
SS = 4  # supersampling factor

PAPER_HI, PAPER_LO = (247, 240, 224), (228, 213, 178)
HALO = (243, 234, 214, 255)
INK_BLUE = (31, 58, 95, 255)
HAT = (20, 28, 43, 255)
HAT_EDGE = (52, 72, 106, 255)
GOLD = (212, 168, 75, 255)
GOLD_DARK = (138, 98, 18, 255)

CENTER = (0.5, 0.52)  # sigil centre
MAC_INSET = 100 / 1024  # macOS icon grid: 824pt body inside a 1024pt canvas


class Pen:
    """Draws ink-like strokes (round-capped stamps with slightly varying width) in unit coordinates."""

    def __init__(self, draw: ImageDraw.ImageDraw, origin: tuple[float, float], span: float):
        self.d, self.ox, self.oy, self.span = draw, origin[0], origin[1], span

    def px(self, u: float, v: float) -> tuple[float, float]:
        return self.ox + (u - 0.5) * self.span, self.oy + (v - 0.5) * self.span

    def poly(self, pts, fill) -> None:
        self.d.polygon([self.px(*p) for p in pts], fill=fill)

    def outline(self, pts, width: float, fill) -> None:
        q = [self.px(*p) for p in pts]
        self.d.line(q + [q[0]], fill=fill, width=max(1, round(width * self.span)), joint="curve")
        r = width * self.span / 2
        for x, y in q:
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def stamp(self, u: float, v: float, w: float, fill=INK_BLUE) -> None:
        x, y = self.px(u, v)
        r = w * self.span / 2
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def stroke(self, pts, w: float, fill=INK_BLUE, taper: bool = True) -> None:
        """Polyline stamped densely; width swells mid-stroke when taper is set."""
        segs = list(zip(pts, pts[1:]))
        total = sum(math.dist(a, b) for a, b in segs) or 1e-9
        step = max(w * 0.3, 1e-4)
        done = 0.0
        for a, b in segs:
            ln = math.dist(a, b)
            n = max(1, int(ln / step))
            for i in range(n + 1):
                t = (done + ln * i / n) / total
                ww = w * (0.65 + 0.35 * math.sin(math.pi * t)) if taper else w
                self.stamp(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, ww, fill)
            done += ln

    def ring(self, r: float, w: float, phase: float = 0.0, fill=INK_BLUE) -> None:
        """Closed ring with hand-drawn wobble in radius and weight."""
        n = max(360, int(2 * math.pi * r / (w * 0.3)))
        for i in range(n + 1):
            th = 2 * math.pi * i / n
            wob = 1 + 0.004 * math.sin(3 * th + phase) + 0.003 * math.sin(7 * th + 2 * phase)
            ww = w * (1 + 0.22 * math.sin(2 * th + phase) + 0.1 * math.sin(5 * th))
            self.stamp(CENTER[0] + r * wob * math.cos(th), CENTER[1] + r * wob * math.sin(th), ww)

    def sparkle(self, cx: float, cy: float, r: float, fill, edge=None, edge_w: float = 0.0) -> None:
        k = r * 0.2
        pts = [(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k), (cx, cy + r), (cx - k, cy + k), (cx - r, cy), (cx - k, cy - k)]
        if edge:
            self.outline(pts, edge_w, edge)
        self.poly(pts, fill)


def polar(r: float, th: float) -> tuple[float, float]:
    return CENTER[0] + r * math.cos(th), CENTER[1] + r * math.sin(th)


def sigil(pen: Pen, detail: int) -> None:
    if detail == 0:
        pen.ring(0.40, 0.05)
        return
    pen.ring(0.40, 0.017, 0.0)
    pen.ring(0.33, 0.008, 1.3)
    if detail == 1:
        for i in range(24):
            th = 2 * math.pi * i / 24
            pen.stroke([polar(0.345, th), polar(0.385, th)], 0.009, taper=False)
        return
    n = 48
    for i in range(n):
        th = 2 * math.pi * i / n - math.pi / 2
        if i % 6 in (2, 3, 4):
            continue  # room for a rune
        long = i % 6 == 0
        pen.stroke([polar(0.338 if long else 0.348, th), polar(0.392 if long else 0.382, th)], 0.0055 if long else 0.004)
    for k in range(8):
        th = 2 * math.pi * (k * 6 + 3) / n - math.pi / 2
        rune(pen, th, k % 4)


def rune(pen: Pen, th: float, kind: int) -> None:
    """Small original marks (dot, cross, arc, chevron) placed in the ring band, oriented to the radius."""
    r0 = 0.365
    rad = (math.cos(th), math.sin(th))
    tan = (-rad[1], rad[0])

    def at(a: float, b: float) -> tuple[float, float]:
        return CENTER[0] + (r0 + b) * rad[0] + a * tan[0], CENTER[1] + (r0 + b) * rad[1] + a * tan[1]

    w = 0.006
    if kind == 0:
        pen.stamp(*at(0, 0), 0.016, INK_BLUE)
        pen.stroke([at(0, -0.022), at(0, -0.012)], w * 0.8)
    elif kind == 1:
        pen.stroke([at(0, -0.022), at(0, 0.022)], w)
        pen.stroke([at(-0.013, 0.004), at(0.013, 0.004)], w * 0.8)
    elif kind == 2:
        pen.stroke([at(0.015 * math.cos(p), 0.002 - 0.015 * math.sin(p)) for p in [math.pi * i / 12 for i in range(13)]], w)
        pen.stamp(*at(0, 0.02), 0.009, INK_BLUE)
    else:
        pen.stroke([at(-0.014, -0.019), at(0, 0.017), at(0.014, -0.019)], w)


def bezier(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    m = 1 - t
    return tuple(m**3 * a + 3 * m * m * t * b + 3 * m * t * t * c + t**3 * d for a, b, c, d in zip(p0, p1, p2, p3))


SPINE = ((0.50, 0.66), (0.485, 0.46), (0.48, 0.30), (0.71, 0.17))  # crooked cone: rises, then bends right at the tip
BASE_HALF = 0.14
BASE_BULGE = 0.04


def spine_edges(t0: float, t1: float, n: int = 60):
    """Both edges of the cone between spine parameters t0..t1."""
    left, right = [], []
    for i in range(n + 1):
        t = t0 + (t1 - t0) * i / n
        x, y = bezier(*SPINE, t)
        x2, y2 = bezier(*SPINE, min(1.0, t + 1e-3))
        x1, y1 = bezier(*SPINE, max(0.0, t - 1e-3))
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy) or 1
        nx, ny = -dy / ln, dx / ln
        hw = BASE_HALF * (1 - t) ** 0.9
        left.append((x - nx * hw, y - ny * hw))
        right.append((x + nx * hw, y + ny * hw))
    return left, right


def hat(pen: Pen, detail: int) -> None:
    halo = 0.034 if detail == 0 else 0.02
    # Brim: wide ellipse tilted a few degrees.
    tilt = math.radians(-5)
    brim = []
    for i in range(240):
        a = 2 * math.pi * i / 240
        ex, ey = 0.32 * math.cos(a), 0.075 * math.sin(a)
        brim.append((0.5 + ex * math.cos(tilt) - ey * math.sin(tilt), 0.665 + ex * math.sin(tilt) + ey * math.cos(tilt)))
    left, right = spine_edges(0.0, 1.0)
    arc = [(0.5 + BASE_HALF * math.cos(p), 0.66 + BASE_BULGE * math.sin(p)) for p in [math.pi * (1 - i / 24) for i in range(25)]]
    cone = right + left[::-1] + arc  # right edge up to the tip, left edge down, then the front of the base
    pen.outline(brim, halo * 2, HALO)
    pen.poly(brim, HAT)
    pen.outline(cone, halo * 2, HALO)
    pen.poly(brim, HAT)
    pen.poly(cone, HAT)
    if detail >= 1:
        pen.stroke(arc, 0.007, HAT_EDGE, taper=True)  # where the cone meets the brim
    # Band above the brim.
    bl, br = spine_edges(0.10, 0.25, 12)
    pen.poly(br + bl[::-1], GOLD)
    if detail >= 2:
        cx, cy = bezier(*SPINE, 0.175)
        nib = [(0, 0.026), (-0.014, -0.004), (-0.009, -0.021), (0.009, -0.021), (0.014, -0.004)]
        pen.poly([(cx + x, cy + y) for x, y in nib], HAT)
        pen.stroke([(cx, cy + 0.02), (cx, cy - 0.006)], 0.0035, GOLD, taper=False)
        pen.stamp(cx, cy - 0.011, 0.007, GOLD)


def star(pen: Pen, detail: int) -> None:
    cx, cy, r = 0.29, 0.30, 0.075
    pen.stamp(cx, cy, r * 2.3, HALO)
    pen.sparkle(cx, cy, r, GOLD, GOLD_DARK if detail >= 2 else None, 0.006)
    if detail >= 1:
        pen.stamp(0.77, 0.35, 0.07, HALO)
        pen.sparkle(0.77, 0.35, 0.028, GOLD, GOLD_DARK if detail >= 2 else None, 0.004)


def detail_for(size: int) -> int:
    return 0 if size <= 48 else 1 if size <= 256 else 2


def paper(size: int) -> Image.Image:
    mask = Image.radial_gradient("L").resize((size, size), Image.BILINEAR)
    return Image.composite(Image.new("RGBA", (size, size), PAPER_LO + (255,)), Image.new("RGBA", (size, size), PAPER_HI + (255,)), mask)


def icon(size: int, *, shape: str) -> Image.Image:
    """shape: 'round' (transparent corners, full canvas), 'full' (opaque square, art in safe zone), 'mac' (inset body with shadow)."""
    big = size * SS
    inset = round(big * MAC_INSET) if shape == "mac" else 0
    body = big - 2 * inset
    layer = paper(body)
    art = Image.new("RGBA", (body, body), (0, 0, 0, 0))
    span = body * (0.8 if shape == "full" else 1.0)
    pen = Pen(ImageDraw.Draw(art), (body / 2, body / 2), span)
    detail = detail_for(size)
    sigil(pen, detail)
    hat(pen, detail)
    star(pen, detail)
    layer.alpha_composite(art)
    if shape != "full":
        mask = Image.new("L", (body * 2, body * 2), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, body * 2 - 1, body * 2 - 1], radius=body * 2 * 0.225, fill=255)
        layer.putalpha(mask.resize((body, body), Image.LANCZOS))
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    if shape == "mac":
        shadow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        shadow.paste((20, 28, 43, 110), (inset, inset + round(big * 0.01)), layer.getchannel("A"))
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(big * 0.012)))
    canvas.alpha_composite(layer, (inset, inset))
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    icon(192, shape="round").save(OUT / "icon-192.png")
    icon(512, shape="round").save(OUT / "icon-512.png")
    # Maskable: full bleed, art inside the central 80% safe zone.
    icon(512, shape="full").save(OUT / "maskable-512.png")
    # iOS applies its own corner mask and dislikes transparency.
    icon(180, shape="full").convert("RGB").save(OUT / "apple-touch-icon.png")
    icon(32, shape="round").save(OUT / "favicon-32.png")
    # Mac app icon source: body inset on the macOS grid with a soft shadow.
    icon(1024, shape="mac").save(OUT / "icon-1024.png")


if __name__ == "__main__":
    main()
