"""Per-project art style presets and the pure functions that add a style to a prompt. See API.md ("Art styles")."""

from __future__ import annotations

NONE_ID = "none"
CUSTOM_ID = "custom"
CUSTOM_MAX_CHARS = 300


def _p(id: str, label: str, description: str, snippet: str, keep: str) -> dict:
    return {"id": id, "label": label, "description": description, "snippet": snippet, "keep": keep}


PRESETS: list[dict] = [
    _p(NONE_ID, "None", "No style added; the prompt is sent as typed.", "", ""),
    _p("photoreal", "Photoreal", "A sharp, natural-light photograph with lifelike texture.",
       "as a photorealistic photograph, shot on a full-frame camera with an 85mm lens, natural light, sharp focus and lifelike texture",
       "Keep the photorealistic photographic style."),
    _p("cinematic", "Cinematic", "A film still with dramatic light and a graded look.",
       "in a cinematic style, like a film still with dramatic lighting, shallow depth of field and rich color grading",
       "Keep the cinematic film-still style."),
    _p("anime", "Anime / cel-shaded", "Clean line art and flat, cel-shaded colors.",
       "in a cel-shaded anime illustration style, clean line art, flat colors",
       "Keep the cel-shaded anime illustration style."),
    _p("stylized-anime", "Stylized anime", "Heavy black ink, hard shadows and wild perspective, like a cult action anime film.",
       "in a stylized anime film style with extremely heavy black ink outlines, large solid-black shadow shapes, exaggerated "
       "perspective and foreshortening, angular hyper-dynamic poses, saturated high-contrast colors and gritty hand-drawn "
       "texture, like a 2000s cult action anime",
       "Keep the stylized anime style with heavy black ink outlines and solid-black shadows."),
    _p("hand-drawn", "Hand-drawn animation", "Classic 2D animation cels with soft pencil outlines and painted backgrounds.",
       "in a classic hand-drawn 2D animation style, like a 1990s animated feature cel, soft pencil-textured outlines, gouache-painted background",
       "Keep the hand-drawn 2D animation style."),
    _p("watercolor", "Watercolor", "Soft washes of color on textured paper.",
       "in a watercolor painting style, soft translucent washes, bleeding edges and visible paper texture",
       "Keep the watercolor painting style."),
    _p("oil-painting", "Oil painting", "Rich brush strokes and heavy paint on canvas.",
       "in an oil painting style, thick visible brush strokes, rich pigment and canvas texture",
       "Keep the oil painting style."),
    _p("pixel-art", "Pixel art", "Retro 16-bit sprites with large visible pixels and a limited palette.",
       "as retro 16-bit pixel art, a low-resolution sprite with large visible square pixels, dithering and a limited color palette",
       "Keep the pixel art style."),
    _p("3d-render", "3D render", "A polished computer-generated render.",
       "as a 3D render, smooth shaded surfaces, physically based materials and soft studio lighting",
       "Keep the 3D render style."),
    _p("low-poly", "Low-poly", "Faceted geometry made of flat triangles.",
       "in a low-poly 3D style, faceted geometry made of flat-shaded triangles and simple colors",
       "Keep the low-poly 3D style."),
    _p("ink-line-art", "Ink line art", "Black ink outlines and hatching on white.",
       "as ink line art, black pen outlines with cross-hatching on a plain white background, no color fills",
       "Keep the ink line art style."),
    _p("comic-book", "Comic book", "Bold outlines, halftone dots and punchy colors.",
       "in a comic book style, bold ink outlines, halftone dots and punchy saturated colors",
       "Keep the comic book style."),
    _p(CUSTOM_ID, "Custom", "Describe the style in your own words.", "", ""),
]
BY_ID: dict[str, dict] = {p["id"]: p for p in PRESETS}


def list_presets() -> list[dict]:
    """The presets as served by GET /api/styles (the custom entry is marked `custom: true`)."""
    return [{**p, "custom": True} if p["id"] == CUSTOM_ID else dict(p) for p in PRESETS]


def normalize(style) -> dict:
    """The stored form `{id}` or `{id: "custom", custom: text}`. Missing/unknown/empty-custom input becomes `{id: "none"}`."""
    if not isinstance(style, dict) or style.get("id") not in BY_ID:
        return {"id": NONE_ID}
    if style["id"] == CUSTOM_ID:
        text = style.get("custom")
        text = text.strip() if isinstance(text, str) else ""
        return {"id": CUSTOM_ID, "custom": text} if text else {"id": NONE_ID}
    return {"id": style["id"]}


def same(a, b) -> bool:
    """Whether two stored styles are the same (id, and the text for custom)."""
    return normalize(a) == normalize(b)


def validate(style) -> dict:
    """Validate a style from a request and return its stored form. Raises ValueError with a client-facing message."""
    if not isinstance(style, dict):
        raise ValueError("style must be an object {id, custom?}")
    extra = set(style) - {"id", "custom"}
    if extra:
        raise ValueError(f"unknown style field(s): {sorted(extra)}")
    sid = style.get("id")
    if not isinstance(sid, str) or sid not in BY_ID:
        raise ValueError(f"unknown style id {sid!r}; expected one of {[p['id'] for p in PRESETS]}")
    text = style.get("custom")
    if sid != CUSTOM_ID:
        if text is not None:
            raise ValueError("custom text is only allowed with style id 'custom'")
        return {"id": sid}
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a custom style needs non-empty custom text")
    text = text.strip()
    if len(text) > CUSTOM_MAX_CHARS:
        raise ValueError(f"custom style text must be at most {CUSTOM_MAX_CHARS} characters")
    return {"id": CUSTOM_ID, "custom": text}


def resolve(style) -> dict:
    """`{id, label, custom?, snippet, keep}` for a stored style."""
    s = normalize(style)
    preset = BY_ID[s["id"]]
    out = {"id": s["id"], "label": preset["label"], "snippet": preset["snippet"], "keep": preset["keep"]}
    if s["id"] == CUSTOM_ID:
        out["custom"] = s["custom"]
        out["snippet"] = s["custom"]
        out["keep"] = f"Keep the same art style: {s['custom']}"
    return out


def record(style) -> dict:
    """What a run keeps of its style: `{id, label, custom?}`."""
    r = resolve(style)
    return {k: r[k] for k in ("id", "label", "custom") if k in r}


def _is_edit(mode: str) -> bool:
    return mode in ("edit", "i2i")


def compose(prompt: str, style, mode: str, *, skip: bool = False) -> str:
    """The prompt with the style added: new image `<prompt>. <snippet>`, edit `<prompt>. <keep>` (whitespace trimmed and one trailing
    period of the prompt dropped). Unchanged for style none or `skip`."""
    r = resolve(style)
    add = r["keep"] if _is_edit(mode) else r["snippet"]
    if skip or not add:
        return prompt
    p = prompt.strip()
    if p.endswith("."):
        p = p[:-1]
    return f"{p}. {add}"


def enhancer_hint(style, mode: str) -> str:
    """Text appended to the user's prompt for the prompt enhancer: `\\n\\nArt style: <snippet>` (new image) or `\\n\\n<keep>` (edit)."""
    r = resolve(style)
    if _is_edit(mode):
        return f"\n\n{r['keep']}" if r["keep"] else ""
    return f"\n\nArt style: {r['snippet']}" if r["snippet"] else ""
