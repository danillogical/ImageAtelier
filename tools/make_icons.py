"""Generate the PWA / app icons into imageatelier/static/icons/ (run: .venv/bin/python tools/make_icons.py).

Art: a witch's drawing pen laid diagonally across a cream title-plate diamond, on a twilight sky with gold art-deco corners.
All geometry is in unit coordinates (0..1) and rendered at 4x, then downscaled with LANCZOS.
"""

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

OUT = Path(__file__).resolve().parent.parent / "imageatelier" / "static" / "icons"
SS = 4  # supersampling factor

INK = (42, 31, 26, 255)
CREAM = (250, 241, 218, 255)
CREAM_SHADE = (224, 205, 168, 255)
PARCHMENT = (246, 236, 212, 255)
PARCHMENT_SHADE = (232, 217, 184, 255)
GOLD = (184, 153, 90, 255)
GOLD_HI = (240, 214, 148, 255)
GOLD_BRIGHT = (226, 182, 84, 255)
GOLD_SHADE = (166, 120, 48, 255)
TEAL = (58, 154, 160, 255)
TEAL_SHADE = (34, 108, 120, 255)
STEEL = (178, 186, 198, 255)
STEEL_SHADE = (120, 130, 148, 255)

SKY = [  # (v, colour) stops: indigo -> lavender -> peach -> gold glow
    (0.00, (30, 24, 78)),
    (0.35, (82, 62, 140)),
    (0.62, (150, 120, 190)),
    (0.82, (238, 168, 150)),
    (1.00, (252, 208, 128)),
]

SKY_MID = (96, 72, 150, 255)

MAC_INSET =100 / 1024  # macOS icon grid: 824pt body inside a 1024pt canvas

PLATE_H = 0.445  # half-diagonal of the title-plate diamond
PEN_C = (0.5, 0.5)
PEN_L = 1.12  # pen length along its axis (about 91% of the rounded icon's corner-to-corner diagonal)
PEN_A = math.radians(-45)  # axis direction: end cap lower-left, nib upper-right


class Pen:
    """Draws onto an RGBA image in unit coordinates (0..1 maps onto `span` pixels centred on `origin`)."""

    def __init__(self, img: Image.Image, origin: tuple[float, float], span: float):
        self.img, self.d, self.ox, self.oy, self.span = img, ImageDraw.Draw(img), origin[0], origin[1], span

    def px(self, u: float, v: float) -> tuple[float, float]:
        return self.ox + (u - 0.5) * self.span, self.oy + (v - 0.5) * self.span

    def poly(self, pts, fill) -> None:
        self.d.polygon([self.px(*p) for p in pts], fill=fill)

    def outline(self, pts, width: float, fill, closed: bool = True) -> None:
        q = [self.px(*p) for p in pts]
        self.d.line(q + ([q[0]] if closed else []), fill=fill, width=max(1, round(width * self.span)), joint="curve")
        r = width * self.span / 2
        for x, y in q:
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def stamp(self, u: float, v: float, w: float, fill=INK) -> None:
        x, y = self.px(u, v)
        r = w * self.span / 2
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def stroke(self, pts, w: float, fill=INK, taper: bool = True) -> None:
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
                ww = w * (0.55 + 0.45 * math.sin(math.pi * t)) if taper else w
                self.stamp(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, ww, fill)
            done += ln

    def line(self, a, b, w: float, fill) -> None:
        self.d.line([self.px(*a), self.px(*b)], fill=fill, width=max(1, round(w * self.span)))

    def glow(self, u: float, v: float, r: float, color) -> None:
        """Soft radial glow of radius r (unit coords) with peak alpha color[3]."""
        x, y = self.px(u, v)
        rad = max(2, round(r * self.span))
        g = Image.radial_gradient("L").resize((2 * rad, 2 * rad), Image.BILINEAR)
        alpha = g.point(lambda p: round(color[3] * max(0.0, 1 - p / 170) ** 2))
        tile = Image.new("RGBA", (2 * rad, 2 * rad), color[:3] + (0,))
        tile.putalpha(alpha)
        self.img.alpha_composite(tile, (round(x - rad), round(y - rad)))

    def sparkle(self, cx: float, cy: float, r: float, fill, edge=None, edge_w: float = 0.0) -> None:
        """Four-point sparkle with pinched waist."""
        k = r * 0.2
        pts = [(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k), (cx, cy + r), (cx - k, cy + k), (cx - r, cy), (cx - k, cy - k)]
        if edge:
            self.outline(pts, edge_w, edge)
        self.poly(pts, fill)


def lerp(a, b, t: float):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def detail_for(size: int) -> int:
    return 0 if size <= 48 else 1 if size <= 256 else 2


# ---------------------------------------------------------------- background


def sky(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):
        v = y / (size - 1)
        for (v0, c0), (v1, c1) in zip(SKY, SKY[1:]):
            if v <= v1:
                col = lerp(c0, c1, (v - v0) / (v1 - v0))
                break
        d.line([(0, y), (size, y)], fill=col + (255,))
    return img


def clouds(img: Image.Image, detail: int) -> None:
    """A faint, soft cloud band low in the sky."""
    size = img.width
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    puffs = [(0.10, 0.86, 0.20, 0.045), (0.30, 0.90, 0.22, 0.04), (0.55, 0.88, 0.2, 0.04), (0.78, 0.85, 0.22, 0.045), (0.95, 0.9, 0.18, 0.04)]
    for u, v, rx, ry in puffs:
        d.ellipse([(u - rx) * size, (v - ry) * size, (u + rx) * size, (v + ry) * size], fill=(214, 160, 190, 120))
        d.ellipse([(u - rx * 0.7) * size, (v - ry * 1.3) * size, (u + rx * 0.7) * size, (v + ry * 0.3) * size], fill=(238, 190, 190, 90))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(size * 0.012)))


def sparkles(pen: Pen, detail: int) -> None:
    """A few sparkles in the sky pockets beside the plate tips."""
    spots = [(0.5, 0.032, 0.022), (0.968, 0.5, 0.016), (0.032, 0.5, 0.013), (0.5, 0.968, 0.017)]
    for u, v, r in spots[: 4 if detail >= 1 else 0]:
        pen.sparkle(u, v, r, CREAM)


# ---------------------------------------------------------------- corner panels

PANEL_D = 0.47  # corner panel hypotenuse: u + v = PANEL_D (parallel to the plate edge, with a gap of sky)
CORNER_R = 0.225  # corner radius of the icon's rounded-rect mask, in unit coords
PANEL_FILL = (44, 32, 86, 130)


def frame_path(k: float) -> list:
    """Rounded-rect outline inset by k hugging the top-left corner: down the left edge, round the arc, along the top edge."""
    reach = PANEL_D - k - 0.04
    r = CORNER_R - k
    pts = [(k, reach), (k, CORNER_R)]
    pts += [(CORNER_R - r * math.cos(t), CORNER_R - r * math.sin(t)) for t in (math.pi / 2 * i / 14 for i in range(1, 14))]
    return pts + [(CORNER_R, k), (reach, k)]


def ornaments(pen: Pen, detail: int) -> None:
    """Dense, framed art-deco panel filling the triangle between each icon corner and the plate edge (same design mirrored)."""
    layer = Image.new("RGBA", pen.img.size, (0, 0, 0, 0))
    lp = Pen(layer, (pen.ox, pen.oy), pen.span)
    tri = Image.new("L", layer.size, 0)
    td = ImageDraw.Draw(tri)
    D = PANEL_D
    corners = [(fu, fv) for fu in (False, True) for fv in (False, True)]

    def m(p, fu, fv):
        return (1 - p[0] if fu else p[0], 1 - p[1] if fv else p[1])

    for fu, fv in corners:
        td.polygon([pen.px(*m(p, fu, fv)) for p in ((0, 0), (D, 0), (0, D))], fill=255)
        lp.poly([m(p, fu, fv) for p in ((0, 0), (D, 0), (0, D))], PANEL_FILL)
    # Clip to the rounded-rect mask (slightly inset) so panels follow the icon's curved corners.
    rr = Image.new("L", layer.size, 0)
    x0, y0 = pen.px(0.012, 0.012)
    x1, y1 = pen.px(0.988, 0.988)
    ImageDraw.Draw(rr).rounded_rectangle([x0, y0, x1, y1], radius=(CORNER_R - 0.012) * pen.span, fill=255)

    frames = [(0.02, 0.0075), (0.05, 0.005), (0.075, 0.0028), (0.10, 0.0045)] if detail >= 2 else [(0.02, 0.007), (0.055, 0.0045)]
    for fu, fv in corners:

        def line(pts, w, col=GOLD):
            lp.stroke([m(p, fu, fv) for p in pts], w, col, taper=False)

        def diamond_at(c, h, col=GOLD):
            lp.poly([m(p, fu, fv) for p in ((c[0] - h, c[1]), (c[0], c[1] - h), (c[0] + h, c[1]), (c[0], c[1] + h))], col)

        for k, w in frames:
            line(frame_path(k), w)
            line(frame_path(k + w * 0.95), w * 0.35, GOLD_HI)
        # Double border parallel to the plate edge.
        line([(0.03, D - 0.03), (D - 0.03, 0.03)], 0.006)
        line([(0.045, D - 0.045 - 0.012), (D - 0.045 - 0.012, 0.045)], 0.0028, GOLD_HI)
        # Sunburst of fine rays from the inner corner, with nested arcs and a diamond at the hub.
        v = frames[-1][0] + 0.03
        n = 9 if detail >= 2 else 5
        reach = D - 0.05
        for i in range(n):
            th = math.radians(6 + i * 78 / (n - 1))
            c, s_ = math.cos(th), math.sin(th)
            t_end = (reach - 2 * v) / (c + s_)
            line([(v + 0.026 * c, v + 0.026 * s_), (v + t_end * c, v + t_end * s_)], 0.0024 if detail >= 2 else 0.003)
            if detail >= 2:
                lp.stamp(*m((v + t_end * c, v + t_end * s_), fu, fv), 0.0075, GOLD_HI)
        if detail >= 2:
            for r_, w_ in ((0.055, 0.003), (0.09, 0.0025)):
                line([(v + r_ * math.cos(th), v + r_ * math.sin(th)) for th in (math.radians(4 + 82 * i / 20) for i in range(21))], w_)
            for k, _ in frames[1::2]:
                c = CORNER_R - (CORNER_R - k) * math.sqrt(0.5)
                diamond_at((c, c), 0.0105, GOLD_HI)
            diamond_at((v, v), 0.014, GOLD)
    layer.putalpha(ImageChops.multiply(ImageChops.multiply(layer.getchannel("A"), tri), rr))
    pen.img.alpha_composite(layer)


# ---------------------------------------------------------------- title plate


def diamond(h: float):
    return [(0.5, 0.5 - h), (0.5 + h, 0.5), (0.5, 0.5 + h), (0.5 - h, 0.5)]


def plate(pen: Pen, detail: int) -> None:
    H = PLATE_H
    ow = 0.012 if detail == 0 else 0.0075
    pen.poly(diamond(H + 0.004), INK)
    pen.poly(diamond(H), GOLD)
    if detail == 0:
        # At favicon size the cream pen would vanish on cream: keep a gold diamond outline over the twilight instead.
        pen.poly(diamond(H - 0.035), SKY_MID)
        return
    pen.poly(diamond(H - 0.011), PARCHMENT)
    pen.poly(diamond(H - 0.021), GOLD)
    pen.poly(diamond(H - 0.031), PARCHMENT)
    # Highlight on the two upper-left outer edges, cel shading band on the two lower-right inner edges.
    top, right, bottom, left = diamond(H - 0.004)
    pen.outline([left, top], 0.003, GOLD_HI, closed=False)
    t, r, b, _ = diamond(H - 0.031)
    t2, r2, b2, _ = diamond(H - 0.085)
    pen.poly([t, r, b, b2, r2, t2], PARCHMENT_SHADE)
    pen.outline(diamond(H + 0.004), ow * 0.6, INK)
    pen.outline(diamond(H - 0.031), ow * 0.5, GOLD_SHADE)


# ---------------------------------------------------------------- the pen


def xf(x: float, y: float) -> tuple[float, float]:
    """Pen-local (x along the axis 0..1 from end cap to nib tip, y across it, both in pen-length units) -> unit coords."""
    dx, dy = math.cos(PEN_A), math.sin(PEN_A)
    px, py = (x - 0.5) * PEN_L, y * PEN_L
    return PEN_C[0] + px * dx - py * dy, PEN_C[1] + px * dy + py * dx


# Pen layout is authored in reference pixels (x from the nib tip at 25 to the cap tip at 795, half-widths across).
def PX(px: float) -> float:
    return 1 - (px - 25) / 770


def HW(px: float) -> float:
    return px / 770 * 1.07  # widened so the barrel is ~20% of the pen length


def spline(knots, per: int = 14):
    """Catmull-Rom curve through (x, half-width) knots."""
    k = [knots[0]] + list(knots) + [knots[-1]]
    out = []
    for i in range(1, len(k) - 2):
        p0, p1, p2, p3 = k[i - 1], k[i], k[i + 1], k[i + 2]
        for j in range(per):
            t = j / per
            out.append(tuple(0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t**3) for a, b, c, d in zip(p0, p1, p2, p3)))
    return out + [knots[-1]]


# Barrel half-width (reference px) = linear base + two Gaussian swells: exactly two maxima and one minimum, no other wiggles.
BARREL_X0, BARREL_X1 = 140, 470
BARREL_P = (24.8, 42.0, 21.1, 218.1, 40.3, 34.5, 378.5, 43.8)  # base0, base1, A1, c1, s1, A2, c2, s2


def barrel_hw(x: float) -> float:
    b0, b1, a1, c1, s1, a2, c2, s2 = BARREL_P
    base = b0 + (b1 - b0) * (x - BARREL_X0) / (BARREL_X1 - BARREL_X0)
    return base + a1 * math.exp(-0.5 * ((x - c1) / s1) ** 2) + a2 * math.exp(-0.5 * ((x - c2) / s2) ** 2)


def barrel_profile(n: int = 160):
    return [(BARREL_X0 + (BARREL_X1 - BARREL_X0) * i / n, barrel_hw(BARREL_X0 + (BARREL_X1 - BARREL_X0) * i / n)) for i in range(n + 1)]


BARREL_PEAK = max(barrel_profile(660), key=lambda p: p[1])[0]  # widest point of the large swell


def barrel_sign_changes() -> int:
    """Number of sign changes of d(half-width)/dx along the barrel (expect 3: max, min, max)."""
    h = [hw for _, hw in barrel_profile(660)]
    d = [b - a for a, b in zip(h, h[1:])]
    return sum(1 for a, b in zip(d, d[1:]) if a * b < 0)


def bullet(x0: float, x1: float, hw: float):
    pts = [(x0, hw)]
    for i in range(1, 17):
        th = math.pi / 2 * i / 16
        pts.append((x0 + (x1 - x0) * math.sin(th), hw * math.cos(th) ** 0.85))
    return pts


# (name, top-half outline as (px, half-width px) from nib side to cap side, fill)
PARTS = [
    ("nib", [(25, 0), (55, 8.5), (88, 17), (104, 6), (152, 5.5)], STEEL),
    ("barrel", barrel_profile(), CREAM),
    ("grip", spline([(481, 46), (530, 40), (600, 35), (672, 34)]), TEAL),
    ("ringA", [(461, 50), (483, 50)], GOLD_BRIGHT),
    ("ringB", [(669, 39), (689, 39)], GOLD_BRIGHT),
    ("cream", [(687, 28), (742, 28)], CREAM),
    ("ringC", [(739, 32), (755, 32)], GOLD_BRIGHT),
    ("cap", bullet(753, 796, 29), GOLD_BRIGHT),
]


def part_polygon(top) -> list:
    """Mirror a top-half outline about the axis into a closed polygon in unit coords (exactly symmetric)."""
    up = [xf(PX(x), -HW(h)) for x, h in top]
    down = [xf(PX(x), HW(h)) for x, h in reversed(top)]
    return up + down


def star8(pen: Pen, x: float, r: float) -> None:
    pts = []
    for i in range(16):
        th = math.pi * i / 8
        rad = (r if i % 4 == 0 else r * 0.7) if i % 2 == 0 else r * 0.34
        pts.append(xf(x + rad * math.cos(th), rad * math.sin(th)))
    pen.poly(pts, GOLD_BRIGHT)
    pen.outline(pts, 0.0022, GOLD_SHADE)


def inlay_diamond(pen: Pen, x: float, hx: float, hy: float, hollow: bool = False) -> None:
    pts = [xf(x - hx, 0), xf(x, -hy), xf(x + hx, 0), xf(x, hy)]
    if hollow:
        pen.outline(pts, 0.0065, GOLD_BRIGHT)
    else:
        pen.poly(pts, GOLD_BRIGHT)
        pen.outline(pts, 0.0022, GOLD_SHADE)


def inlay_dot(pen: Pen, x: float, r: float) -> None:
    pen.stamp(*xf(x, 0), r * 2, GOLD_BRIGHT)


def draw_pen_flat(pen: Pen, detail: int) -> None:
    """Flat-colour pen: one even ink outline around the merged silhouette, part fills, joint lines and gold inlays."""
    ow = 0.022 if detail == 0 else 0.0155
    polys = [(n, part_polygon(t), c) for n, t, c in PARTS]
    # Ink: every part grown by the stroke width; the union of those is the merged silhouette dilated evenly.
    for _, poly, _ in polys:
        pen.poly(poly, INK)
        pen.outline(poly, ow * 2, INK)
    for _, poly, col in polys:
        pen.poly(poly, col)
    # Thin interior lines only at the part joints.
    for n, poly, _ in polys:
        if n != "nib":
            pen.outline(poly, ow * 0.4, INK)
    if detail >= 1:  # centre ridge of the nib
        pen.stroke([xf(PX(25), 0), xf(PX(100), 0)], 0.0035, STEEL_SHADE, taper=False)

    # Gold inlays, centred on the axis: dot, small diamond, star, hollow diamond, dot; grip diamonds; cream-section triangle.
    # The row sits in the large swell with the star at its widest point (~47% of the local width).
    star8(pen, PX(BARREL_PEAK), 0.0474)
    if detail >= 1:
        inlay_diamond(pen, PX(BARREL_PEAK - 49), 0.014, 0.0095)
        inlay_diamond(pen, PX(BARREL_PEAK + 54), 0.017, 0.027, hollow=True)
        for px in (515, 555, 595, 633):
            inlay_diamond(pen, PX(px), 0.017, 0.0105)
        tri = [xf(PX(697), 0), xf(PX(735), -HW(14)), xf(PX(735), HW(14))]
        pen.poly(tri, GOLD_BRIGHT)
        pen.outline(tri, 0.0022, GOLD_SHADE)
    if detail >= 2:
        inlay_dot(pen, PX(BARREL_PEAK - 68), 0.0058)
        inlay_dot(pen, PX(BARREL_PEAK + 73), 0.0062)


def pen_flat_layer(span: float, detail: int, mirror: bool = True) -> Image.Image:
    """The flat pen drawn horizontally (nib right) on a transparent canvas. mirror=True rebuilds the lower half from the upper one.

    PIL's polygon fill rules are not mirror-symmetric at the pixel level, so the mirror step makes the pen exactly symmetric.
    """
    global PEN_A, PEN_C
    saved = PEN_A, PEN_C
    PEN_A, PEN_C = 0.0, (0.5, 0.5)
    try:
        w = round(PEN_L * span * 1.08)
        h = round(0.34 * span) // 2 * 2  # even, so the axis sits between two rows
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_pen_flat(Pen(img, (w / 2, (h - 1) / 2), span), detail)  # axis at row (h-1)/2 so row i pairs with h-1-i
    finally:
        PEN_A, PEN_C = saved
    if mirror:
        img.paste(img.crop((0, 0, w, h // 2)).transpose(Image.FLIP_TOP_BOTTOM), (0, h - h // 2))
    return img


def draw_pen(pen: Pen, detail: int) -> None:
    ow = 0.022 if detail == 0 else 0.0155
    polys = [part_polygon(t) for _, t, _ in PARTS]

    # Soft drop shadow on the plate.
    sh = Image.new("RGBA", pen.img.size, (0, 0, 0, 0))
    sp = Pen(sh, (pen.ox + pen.span * 0.012, pen.oy + pen.span * 0.018), pen.span)
    for poly in polys:
        sp.poly(poly, (60, 30, 50, 120))
        sp.outline(poly, ow * 2, (60, 30, 50, 120))
    pen.img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(pen.span * 0.012)))

    # Exactly symmetric flat pen, rotated onto the icon diagonal (nib to the upper right).
    flat = pen_flat_layer(pen.span, detail).rotate(-math.degrees(PEN_A), resample=Image.BICUBIC, expand=True)
    cx, cy = pen.px(*PEN_C)
    pen.img.alpha_composite(flat, (round(cx - flat.width / 2), round(cy - flat.height / 2)))

    # One shade band along the lower-right side of the whole pen, soft-edged and clipped to the silhouette.
    band = Image.new("L", pen.img.size, 0)
    sil = Image.new("L", pen.img.size, 0)
    bp = Pen(band, (pen.ox, pen.oy), pen.span)
    sp = Pen(sil, (pen.ox, pen.oy), pen.span)
    for n, top, _ in PARTS:
        sp.poly(part_polygon(top), 255)
        if n == "nib":
            continue
        inner = [xf(PX(x), 0.42 * HW(h)) for x, h in top]
        outer = [xf(PX(x), HW(h)) for x, h in reversed(top)]
        bp.poly(inner + outer, 255)
    band = ImageChops.multiply(band.filter(ImageFilter.GaussianBlur(pen.span * 0.004)), sil)
    tint = Image.new("RGBA", pen.img.size, (74, 40, 72, 0))
    tint.putalpha(band.point(lambda v: round(v * 0.36)))
    pen.img.alpha_composite(tint)
    # Thin highlight on the opposite (upper-left) side of the long parts.
    if detail >= 1:
        for n, top, _ in PARTS:
            if n not in ("barrel", "grip", "cap"):
                continue
            col = {"barrel": (255, 253, 242, 255), "grip": (150, 214, 208, 255), "cap": GOLD_HI}[n]
            seg = top[int(len(top) * 0.12): int(len(top) * 0.88)] if n != "cap" else top[3:10]
            pen.stroke([xf(PX(x), -0.74 * HW(h)) for x, h in seg], 0.0045 if detail >= 2 else 0.006, col)


def hero(pen: Pen, detail: int) -> None:
    draw_pen(pen, detail)


def symmetry_check(size: int = 512) -> dict:
    """Mean |RGBA diff| between the unshaded horizontal pen and its top-to-bottom mirror, before and after the mirror step."""
    out = {}
    for mirror in (False, True):
        img = pen_flat_layer(size * SS * 0.85, 2, mirror)
        small = img.resize((img.width // SS, img.height // SS), Image.LANCZOS)
        diff = ImageChops.difference(small, small.transpose(Image.FLIP_TOP_BOTTOM))
        out["mirrored" if mirror else "raw"] = (sum(ImageStat.Stat(diff).mean) / 4, sum(ImageStat.Stat(diff.crop(small.getchannel("A").getbbox())).mean) / 4)
    return out


def trail(pen: Pen, detail: int) -> None:
    """A short flowing ink stroke with a glowing sparkle and a few tiny dots, leaving the nib tip."""
    if detail == 0:
        return
    tip = xf(1.0, 0)
    dx, dy = math.cos(PEN_A), math.sin(PEN_A)
    nx, ny = -dy, dx
    pts = []
    for i in range(33):
        t = i / 32
        a = 0.008 + 0.022 * t
        b = 0.008 * math.sin(2 * math.pi * 1.1 * t) * t
        pts.append((tip[0] + dx * a + nx * b, tip[1] + dy * a + ny * b))
    end = pts[-1]
    pen.glow(end[0] + dx * 0.012, end[1] + dy * 0.012, 0.06, (255, 232, 160, 190))
    pen.stroke(pts, 0.008, GOLD_HI)
    sx, sy = end[0] + dx * 0.014, end[1] + dy * 0.014
    pen.sparkle(sx, sy, 0.019, CREAM, GOLD_BRIGHT, 0.003)
    if detail >= 2:
        for k, (da, db, w) in enumerate(((0.0, -0.03, 0.006), (0.0, 0.03, 0.005))):
            pen.stamp(end[0] + dx * da + nx * db, end[1] + dy * da + ny * db, w, CREAM if k != 1 else GOLD_HI)


def icon(size: int, *, shape: str) -> Image.Image:
    """shape: 'round' (transparent corners, full canvas), 'full' (opaque square, art in safe zone), 'mac' (inset body with shadow)."""
    big = size * SS
    inset = round(big * MAC_INSET) if shape == "mac" else 0
    body = big - 2 * inset
    layer = sky(body)
    detail = detail_for(size)
    span = body * (0.8 if shape == "full" else 1.0)
    pen = Pen(layer, (body / 2, body / 2), span)
    if detail >= 1:
        clouds(layer, detail)
        pen = Pen(layer, (body / 2, body / 2), span)
    sparkles(pen, detail)
    if detail >= 1:
        ornaments(pen, detail)
    plate(pen, detail)
    hero(pen, detail)
    trail(pen, detail)
    if shape != "full":
        mask = Image.new("L", (body * 2, body * 2), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, body * 2 - 1, body * 2 - 1], radius=body * 2 * 0.225, fill=255)
        layer.putalpha(mask.resize((body, body), Image.LANCZOS))
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    if shape == "mac":
        shadow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        shadow.paste((20, 14, 43, 110), (inset, inset + round(big * 0.01)), layer.getchannel("A"))
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
    print(f"barrel half-width derivative sign changes: {barrel_sign_changes()} (max {BARREL_PEAK:.0f}px)")
    for k, (full, crop) in symmetry_check().items():
        print(f"pen symmetry, {k}: mean abs RGBA diff vs top-bottom mirror = {full:.4f} (whole canvas), {crop:.4f} (pen bbox)")


if __name__ == "__main__":
    main()
