"""Persistent user settings (data/settings.json) with validation."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path

from imageatelier import models, suites

log = logging.getLogger(__name__)

DEFAULTS: dict = {
    "model_suite": "q8",
    "steps": 40,
    "low_memory": True,
    "vae_tiling": False,
    "resolution": 1024,
    "pe_thinking": True,
    "pe_max_new_tokens": 8192,  # edit (I2I) rewrites can think ~4k tokens before answering
    "pe_language": "english",
    "ref_detail": "auto",  # reference_resolution for extra refs: auto (by count) | high 1024 | standard 768 | low 512
    "max_refs": 3,
}
REF_DETAILS = ("auto", "high", "standard", "low")
MAX_REFS_CAP = 9  # the model takes at most 10 images: the current one plus 9 references
PE_VARIANTS = ("official",)
PE_QUANTS = ("bf16", "8bit", "6bit", "4bit")
PE_LANGUAGES = ("english", "auto")
MIN_PE_TOKENS, MAX_PE_TOKENS = 256, 16384
NON_TURBO_DEFAULT_STEPS = 40
MIN_STEPS, MAX_STEPS = 2, 100
MIN_RESOLUTION, MAX_RESOLUTION = 256, 2048


def default_steps(turbo: bool) -> int:
    return DEFAULTS["steps"] if turbo else NON_TURBO_DEFAULT_STEPS


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_steps(steps, turbo: bool) -> None:
    if not _is_int(steps):
        raise ValueError(f"steps must be an integer, got {steps!r}")
    if turbo:
        raise ValueError("Turbo is no longer supported")
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}, got {steps}")


def validate(s: dict) -> None:
    """Raise ValueError if a complete settings dict is invalid."""
    if not isinstance(s["model_suite"], str) or s["model_suite"] not in suites.VARIANTS:
        raise ValueError(f"model_suite must be one of {list(suites.VARIANTS)}")
    if s["pe_language"] not in PE_LANGUAGES:
        raise ValueError(f"pe_language must be one of {list(PE_LANGUAGES)}, got {s['pe_language']!r}")
    if not _is_int(s["pe_max_new_tokens"]) or not MIN_PE_TOKENS <= s["pe_max_new_tokens"] <= MAX_PE_TOKENS:
        raise ValueError(f"pe_max_new_tokens must be an integer between {MIN_PE_TOKENS} and {MAX_PE_TOKENS}")
    for key in ("low_memory", "vae_tiling", "pe_thinking"):
        if not isinstance(s[key], bool):
            raise ValueError(f"{key} must be a boolean, got {s[key]!r}")
    if not _is_int(s["resolution"]) or not MIN_RESOLUTION <= s["resolution"] <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be an integer between {MIN_RESOLUTION} and {MAX_RESOLUTION}")
    if s["resolution"] % 32:
        raise ValueError(f"resolution must be a multiple of 32, got {s['resolution']}")
    if s["ref_detail"] not in REF_DETAILS:
        raise ValueError(f"ref_detail must be one of {list(REF_DETAILS)}, got {s['ref_detail']!r}")
    if not _is_int(s["max_refs"]) or not 0 <= s["max_refs"] <= MAX_REFS_CAP:
        raise ValueError(f"max_refs must be an integer between 0 and {MAX_REFS_CAP}")
    validate_steps(s["steps"], False)


class SettingsStore:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._values = dict(DEFAULTS)
        self.migrated: list[str] = []  # keys coerced from 4bit on load (the app logs this once logging is up)
        self.migrated_pe_variant: str | None = None  # a stored "pe_t2i_variant" renamed to pe_variant
        self.migrated_variant: str | None = None  # a stored monolithic "variant" replaced by image_model/text_encoder
        try:
            saved = json.loads(self._path.read_text())
            merged = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
            # Old independent part choices migrate to the official Q8 suite. Keep unrelated preferences.
            legacy = any(k in saved for k in ("image_model", "text_encoder", "turbo", "pe_quant", "pe_variant", "variant"))
            if saved.get("turbo") and saved.get("steps") in (6, 8):
                merged["steps"] = 40
            res = merged["resolution"]
            if isinstance(res, int) and not isinstance(res, bool) and res % 32 and MIN_RESOLUTION <= res <= MAX_RESOLUTION:
                merged["resolution"] = min(MAX_RESOLUTION, max(MIN_RESOLUTION, round(res / 32) * 32))  # older files allowed any value
            validate(merged)
            self._values = merged
            if legacy:
                self._write(merged)
        except (OSError, ValueError, AttributeError):
            pass  # missing, corrupt or invalid file: fall back to defaults

    def _write(self, values: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(values, indent=2))
        os.replace(tmp, self._path)

    @staticmethod
    def _with_parts(values: dict) -> dict:
        quant = models.SUITE_QUANTS[values["model_suite"]]
        return {**values, "image_model": f"official-{quant}", "text_encoder": f"official-{quant}",
                "pe_quant": quant, "pe_variant": "official", "turbo": False}

    def get(self) -> dict:
        with self._lock:
            return self._with_parts(self._values)

    def update(self, partial: dict) -> dict:
        """Apply a partial update; raises ValueError (nothing is changed) if the result is invalid."""
        if not isinstance(partial, dict):
            raise ValueError("settings must be a JSON object")
        if "variant" in partial:
            raise ValueError("variant was replaced by image_model/text_encoder")
        if "pe_t2i_variant" in partial:
            raise ValueError("pe_t2i_variant was replaced by pe_variant")
        unknown = sorted(set(partial) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
        with self._lock:
            new = {**self._values, **partial}
            validate(new)
            self._write(new)
            self._values = new
            return self._with_parts(new)
