"""Output sizes: agreement with the fork's own formula (independent oracle), fixed points, closest(), /api/sizes."""

from __future__ import annotations

import math

import pytest

from imageatelier import sizes

RESOLUTIONS = list(range(256, 2049, 32))
LABELS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
# API.md, "Output size": the presets at resolution 1024.
AT_1024 = {"1:1": (1024, 1024), "4:3": (1184, 896), "3:4": (896, 1184), "3:2": (1248, 832), "2:3": (832, 1248),
           "16:9": (1376, 768), "9:16": (768, 1376)}


@pytest.fixture(scope="module")
def oracle():
    from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator

    return QwenImage21LatentCreator.dimensions


class TestAgainstTheFork:
    @pytest.mark.parametrize("res", RESOLUTIONS)
    def test_dimensions_match_the_fork_for_every_preset(self, oracle, res):
        presets = sizes.presets_for(res)
        assert [p["label"] for p in presets] == LABELS
        for p in presets:
            assert (p["width"], p["height"]) == oracle(res, p["ratio"]), (res, p["label"])
            assert sizes.dimensions(res, p["ratio"]) == oracle(res, p["ratio"])
            assert p["width"] % 32 == 0 and p["height"] % 32 == 0

    @pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 0.9, 1.0, 1.1, 1.3333, 1.7778, 2.4, 5.0, 10.0])
    @pytest.mark.parametrize("res", [256, 512, 1000, 1024, 1536, 2048])
    def test_arbitrary_ratios_match_the_fork(self, oracle, res, ratio):
        assert sizes.dimensions(res, ratio) == oracle(res, ratio)

    def test_documented_sizes_at_1024(self):
        got = {p["label"]: (p["width"], p["height"]) for p in sizes.presets_for(1024)}
        assert got == AT_1024


class TestClosest:
    def ratios(self):
        return {p["label"]: p["ratio"] for p in sizes.presets_for(1024)}

    def test_exact_preset_ratios_map_to_themselves(self):
        for label, (w, h) in AT_1024.items():
            assert sizes.closest(w, h) == label

    @pytest.mark.parametrize("w,h,label", [
        (500, 500, "1:1"), (1, 1, "1:1"), (3024, 4032, "3:4"), (4032, 3024, "4:3"), (1920, 1080, "16:9"),
        (1080, 1920, "9:16"), (3000, 2000, "3:2"), (2000, 3000, "2:3"),
        (100000, 100, "16:9"), (100, 100000, "9:16"), (10000, 1, "16:9"), (1, 10000, "9:16")])
    def test_typical_and_extreme_images(self, w, h, label):
        assert sizes.closest(w, h) == label

    def test_between_two_presets_the_nearer_in_log_ratio_wins(self):
        r = self.ratios()
        ordered = sorted(r, key=lambda k: r[k])
        for lo, hi in zip(ordered, ordered[1:]):
            mid = math.sqrt(r[lo] * r[hi])  # geometric midpoint = equal |log| distance
            for nudge, expect in ((0.99, lo), (1.01, hi)):
                h = 100000
                w = round(mid * nudge * h)
                assert sizes.closest(w, h) == expect, (lo, hi, nudge)

    def test_never_returns_an_unknown_label(self):
        for w, h in [(7, 13), (13, 7), (999, 1000), (1000, 999), (123456, 789)]:
            assert sizes.closest(w, h) in LABELS


class TestApiSizes:
    def test_shape_and_default_resolution(self, client):
        r = client.get("/api/sizes")
        assert r.status_code == 200
        body = r.json()
        assert body["resolution"] == 1024
        assert [p["label"] for p in body["presets"]] == LABELS
        for p in body["presets"]:
            assert set(p) >= {"label", "ratio", "width", "height"}
            assert (p["width"], p["height"]) == AT_1024[p["label"]]
            assert p["ratio"] == pytest.approx(p["width"] / p["height"], rel=0.02)
        assert body["closest"] is None

    @pytest.mark.parametrize("w,h,label", [(3024, 4032, "3:4"), (4032, 3024, "4:3"), (4000, 1000, "16:9"),
                                           (10, 10, "1:1"), (900, 1600, "9:16")])
    def test_closest_pick(self, client, w, h, label):
        body = client.get(f"/api/sizes?w={w}&h={h}").json()
        assert body["closest"] == label

    def test_matches_the_module(self, client):
        assert client.get("/api/sizes").json()["presets"] == sizes.presets_for(1024)

    def test_reflects_resolution_setting(self, client, oracle):
        assert client.put("/api/settings", json={"resolution": 512}).status_code == 200
        body = client.get("/api/sizes?w=3&h=4").json()
        assert body["resolution"] == 512 and body["closest"] == "3:4"
        for p in body["presets"]:
            assert (p["width"], p["height"]) == oracle(512, p["ratio"])
        assert body["presets"][2]["width"] < 896
        client.put("/api/settings", json={"resolution": 1536})
        body = client.get("/api/sizes").json()
        assert body["resolution"] == 1536
        assert {p["label"]: (p["width"], p["height"]) for p in body["presets"]}["1:1"] == (1536, 1536)
