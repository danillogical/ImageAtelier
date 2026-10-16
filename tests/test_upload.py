"""Image upload (start a project from a picture) and the resolution rules, against API.md."""

from __future__ import annotations

import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from imageatelier import sizes
from imageatelier.app import create_app

from conftest import StubEngine, do_run, new_project, submit, wait_run, wait_until
from test_enhance import collect_events

RED, GREEN, BLUE = (220, 20, 20), (20, 200, 20), (20, 20, 220)


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def jpeg(img: Image.Image, exif_orientation: int | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {"quality": 95}
    if exif_orientation is not None:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex.tobytes()
    img.convert("RGB").save(buf, format="JPEG", **kw)
    return buf.getvalue()


def bands(w, h, colors, vertical=False):
    """Image of equal colour bands: left-to-right (or top-to-bottom when vertical)."""
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    n = len(colors)
    for i, c in enumerate(colors):
        if vertical:
            d.rectangle([0, i * h // n, w, (i + 1) * h // n], fill=c)
        else:
            d.rectangle([i * w // n, 0, (i + 1) * w // n, h], fill=c)
    return im


def near(px, want, tol=30):
    return all(abs(a - b) <= tol for a, b in zip(px[:3], want))


def upload(client, pid, data, **params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return client.post(f"/api/projects/{pid}/upload" + (f"?{q}" if q else ""), content=data,
                       headers={"content-type": "application/octet-stream"})


def stored(client, pid, n=1, thumb=False) -> Image.Image:
    r = client.get(f"/api/projects/{pid}/runs/{n}/{'thumb' if thumb else 'image'}")
    assert r.status_code == 200
    im = Image.open(io.BytesIO(r.content))
    im.load()
    return im


def nothing_saved(client, pid, tmp_path):
    p = client.get(f"/api/projects/{pid}").json()
    assert p["runs"] == [] and p["current_run"] is None
    runs = tmp_path / "data" / "projects" / pid / "runs"
    assert not runs.exists() or not any(runs.iterdir())


@pytest.fixture
def pid(client):
    return new_project(client)["id"]


# ---- normalisation --------------------------------------------------------------------------
class TestNormalisation:
    def test_portrait_phone_photo_becomes_3_4(self, client, pid):
        src = bands(3024, 4032, [RED, GREEN, BLUE], vertical=True)
        r = upload(client, pid, jpeg(src))
        assert r.status_code == 201, r.text
        run = r.json()
        assert run["run"] == 1 and run["mode"] == "upload" and run["project_id"] == pid
        assert (run["width"], run["height"]) == (896, 1184) and run["preset"] == "3:4"
        assert run["source_size"] == [3024, 4032]
        x, y, w, h = run["crop"]
        assert 0 <= x and 0 <= y and x + w <= 3024 and y + h <= 4032
        assert w / h == pytest.approx(896 / 1184, abs=0.002)
        assert run["image_url"] == f"/api/projects/{pid}/runs/1/image" and run["thumb_url"].endswith("/runs/1/thumb")
        # generation fields are null, prompt empty
        assert run["prompt"] == "" and run["parent_run"] is None
        for k in ("seed", "steps", "turbo", "image_model", "text_encoder", "gen_s", "load_s", "job_id", "enhanced_from"):
            assert run[k] is None, k
        # saved as an opaque RGB PNG of exactly the output size
        im = stored(client, pid)
        assert im.format == "PNG" and im.mode == "RGB" and im.size == (896, 1184)
        assert near(im.getpixel((448, 20)), RED) and near(im.getpixel((448, 1164)), BLUE)
        thumb = stored(client, pid, thumb=True)
        assert thumb.format == "WEBP" and max(thumb.size) <= 384
        # project state
        p = client.get(f"/api/projects/{pid}").json()
        assert p["current_run"] == 1 and [r["run"] for r in p["runs"]] == [1]
        assert p["runs"][0]["mode"] == "upload" and p["runs"][0]["preset"] == "3:4"
        g = next(x for x in client.get("/api/projects").json() if x["id"] == pid)
        assert g["run_count"] == 1 and g["latest"] is not None

    def test_exif_orientation_6_is_portrait(self, client, pid):
        # stored landscape 4032x3024, left half red / right half blue; orientation 6 displays it rotated 90 CW
        src = bands(4032, 3024, [RED, BLUE])
        r = upload(client, pid, jpeg(src, exif_orientation=6))
        assert r.status_code == 201, r.text
        run = r.json()
        assert run["source_size"] == [3024, 4032]
        assert (run["width"], run["height"]) == (896, 1184) and run["preset"] == "3:4"
        im = stored(client, pid)
        assert near(im.getpixel((448, 20)), RED) and near(im.getpixel((448, 1160)), BLUE)  # left -> top

    def test_without_exif_the_same_pixels_stay_landscape(self, client, pid):
        run = upload(client, pid, jpeg(bands(4032, 3024, [RED, BLUE]))).json()
        assert run["source_size"] == [4032, 3024] and (run["width"], run["height"]) == (1184, 896)
        assert run["preset"] == "4:3"

    def test_panorama_is_cover_cropped_to_16_9(self, client, pid):
        r = upload(client, pid, png(bands(4000, 1000, [RED, GREEN, BLUE])))
        assert r.status_code == 201
        run = r.json()
        assert (run["width"], run["height"]) == (1376, 768) and run["preset"] == "16:9"
        x, y, w, h = run["crop"]
        assert h == 1000 and w == pytest.approx(1000 * 1376 / 768, abs=2) and x == pytest.approx((4000 - w) / 2, abs=2)
        im = stored(client, pid)
        assert im.size == (1376, 768)
        # cover fit: no padding, centred crop keeps the three bands' edges and centre
        assert near(im.getpixel((2, 384)), RED) and near(im.getpixel((688, 384)), GREEN)
        assert near(im.getpixel((1373, 384)), BLUE)
        assert not near(im.getpixel((2, 2)), (255, 255, 255), tol=10)

    def test_tall_strip_goes_to_9_16(self, client, pid):
        run = upload(client, pid, png(bands(300, 3000, [RED, BLUE], vertical=True))).json()
        assert (run["width"], run["height"]) == (768, 1376) and run["preset"] == "9:16"

    def test_square_goes_to_1_1(self, client, pid):
        run = upload(client, pid, png(bands(700, 700, [RED, BLUE]))).json()
        assert (run["width"], run["height"]) == (1024, 1024) and run["preset"] == "1:1"

    def test_image_already_at_a_preset_size_is_kept_as_is(self, client, pid):
        src = Image.new("RGB", (896, 1184))
        px = src.load()
        for yy in range(1184):
            for xx in range(896):
                px[xx, yy] = ((xx * 7) % 256, (yy * 5) % 256, (xx + yy) % 256)  # not resizable without visible change
        r = upload(client, pid, png(src))
        assert r.status_code == 201
        run = r.json()
        assert (run["width"], run["height"]) == (896, 1184) and run["preset"] == "3:4"
        assert run["crop"] == [0, 0, 896, 1184] and run["source_size"] == [896, 1184]
        assert stored(client, pid).tobytes() == src.tobytes()

    def test_transparency_is_flattened_over_white(self, client, pid):
        src = Image.new("RGBA", (896, 1184), (0, 0, 0, 0))  # fully transparent black on the left
        ImageDraw.Draw(src).rectangle([448, 0, 896, 1184], fill=(220, 20, 20, 255))
        run = upload(client, pid, png(src)).json()
        im = stored(client, pid)
        assert im.mode == "RGB" and im.size == (896, 1184) and (run["width"], run["height"]) == (896, 1184)
        assert im.getpixel((100, 600)) == (255, 255, 255)  # transparent pixels are white, not black
        assert im.getpixel((800, 600)) == (220, 20, 20)

    def test_transparency_is_flattened_when_resizing_too(self, client, pid):
        src = Image.new("RGBA", (1000, 1000), (0, 0, 0, 0))
        ImageDraw.Draw(src).rectangle([500, 0, 1000, 1000], fill=(20, 20, 220, 255))
        upload(client, pid, png(src))
        im = stored(client, pid)
        assert im.mode == "RGB" and im.size == (1024, 1024)
        assert near(im.getpixel((100, 500)), (255, 255, 255), tol=4) and near(im.getpixel((900, 500)), BLUE, tol=6)

    def test_palette_and_grayscale_inputs_are_accepted_as_rgb(self, client, pid):
        r = upload(client, pid, png(Image.new("L", (640, 480), 128)))
        assert r.status_code == 201 and stored(client, pid).mode == "RGB"

    def test_resolution_setting_sets_the_output_size(self, client, pid):
        assert client.put("/api/settings", json={"resolution": 512}).status_code == 200
        want = {p["label"]: (p["width"], p["height"]) for p in client.get("/api/sizes").json()["presets"]}
        run = upload(client, pid, jpeg(bands(3024, 4032, [RED, BLUE]))).json()
        assert (run["width"], run["height"]) == want["3:4"] and run["width"] < 896
        assert stored(client, pid).size == want["3:4"]


# ---- explicit preset / crop -----------------------------------------------------------------
class TestPresetAndCrop:
    def test_explicit_crop_is_honoured(self, client, pid):
        src = bands(2000, 1000, [RED, BLUE])
        run = upload(client, pid, png(src), preset="1:1", crop="0,0,1000,1000").json()
        assert run["preset"] == "1:1" and run["crop"] == [0, 0, 1000, 1000]
        assert (run["width"], run["height"]) == (1024, 1024)
        im = stored(client, pid)
        assert near(im.getpixel((10, 10)), RED) and near(im.getpixel((1010, 1010)), RED)  # the left (red) square only

    def test_off_ratio_crop_is_shrunk_around_its_centre(self, client, pid):
        src = bands(2000, 1000, [RED, BLUE])
        run = upload(client, pid, png(src), preset="1:1", crop="0,0,1500,1000").json()
        assert run["crop"] == [250, 0, 1000, 1000]
        im = stored(client, pid)
        assert near(im.getpixel((10, 500)), RED) and near(im.getpixel((1013, 500)), BLUE)

    def test_crop_beyond_the_image_is_clamped(self, client, pid):
        run = upload(client, pid, png(bands(2000, 1000, [RED, BLUE])), preset="1:1", crop="1500,0,1500,1000").json()
        x, y, w, h = run["crop"]
        assert x >= 1500 and y >= 0 and x + w <= 2000 and y + h <= 1000 and w == h and w > 0

    def test_preset_without_crop_uses_a_centred_cover_box(self, client, pid):
        run = upload(client, pid, png(bands(2000, 2000, [RED, GREEN, BLUE], vertical=True)), preset="16:9").json()
        assert (run["width"], run["height"]) == (1376, 768) and run["preset"] == "16:9"
        x, y, w, h = run["crop"]
        assert w == 2000 and h == pytest.approx(2000 * 768 / 1376, abs=2) and y == pytest.approx((2000 - h) / 2, abs=2)
        assert near(stored(client, pid).getpixel((688, 384)), GREEN)

    def test_preset_overrides_the_closest_ratio(self, client, pid):
        run = upload(client, pid, png(bands(4000, 1000, [RED, BLUE])), preset="1:1").json()
        assert run["preset"] == "1:1" and (run["width"], run["height"]) == (1024, 1024)

    def test_explicit_preset_disables_keep_as_is(self, client, pid):
        run = upload(client, pid, png(Image.new("RGB", (896, 1184), RED)), preset="1:1").json()
        assert (run["width"], run["height"]) == (1024, 1024)

    def test_exact_preset_size_upload_keeps_pixels_and_records_crop_as_provenance(self, client, pid):
        src = Image.new("RGB", (896, 1184))
        px = src.load()
        for yy in range(1184):
            for xx in range(896):
                px[xx, yy] = ((xx * 3) % 256, (yy * 11) % 256, (xx * yy) % 256)
        r = upload(client, pid, png(src), preset="3:4", crop="0,0,3024,3996", src="3024x4032")
        assert r.status_code == 201, r.text
        run = r.json()
        assert run["crop"] == [0, 0, 3024, 3996] and run["source_size"] == [3024, 4032] and run["preset"] == "3:4"
        assert (run["width"], run["height"]) == (896, 1184)
        assert stored(client, pid).tobytes() == src.tobytes()  # not re-cropped or resampled

    def test_exact_size_with_crop_but_no_preset_is_also_kept(self, client, pid):
        src = Image.effect_noise((896, 1184), 60).convert("RGB")
        run = upload(client, pid, png(src), crop="10,20,2000,2643", src="3024x4032").json()
        assert run["crop"] == [10, 20, 2000, 2643] and run["preset"] == "3:4"
        assert stored(client, pid).tobytes() == src.tobytes()

    def test_crop_is_applied_when_the_body_is_not_the_preset_size(self, client, pid):
        run = upload(client, pid, png(bands(2000, 1000, [RED, BLUE])), preset="1:1", crop="0,0,1000,1000",
                     src="2000x1000").json()
        assert run["crop"] == [0, 0, 1000, 1000] and (run["width"], run["height"]) == (1024, 1024)
        assert near(stored(client, pid).getpixel((900, 500)), RED)  # re-cropped from the body

    def test_src_records_the_original_oriented_size(self, client, pid):
        run = upload(client, pid, png(Image.new("RGB", (896, 1184), RED)), src="3024x4032").json()
        assert run["source_size"] == [3024, 4032]

    def test_without_src_source_size_is_the_uploaded_size(self, client, pid):
        run = upload(client, pid, png(Image.new("RGB", (896, 1184), RED))).json()
        assert run["source_size"] == [896, 1184]

    @pytest.mark.parametrize("src", ["abc", "100", "0x100", "100x0", "-5x10", "1x2x3", "100x", "x100", "10.5x20", ""])
    def test_malformed_src_400(self, client, pid, tmp_path, src):
        r = upload(client, pid, png(Image.new("RGB", (896, 1184), RED)), src=src)
        assert r.status_code == 400 and "detail" in r.json()
        nothing_saved(client, pid, tmp_path)

    @pytest.mark.parametrize("params", [{"preset": "5:7"}, {"preset": "nope"}, {"preset": "1024x1024"}])
    def test_bad_preset_400(self, client, pid, tmp_path, params):
        r = upload(client, pid, png(bands(800, 600, [RED, BLUE])), **params)
        assert r.status_code == 400 and "detail" in r.json()
        nothing_saved(client, pid, tmp_path)

    @pytest.mark.parametrize("crop", ["abc", "1,2,3", "1,2,3,4,5", "0,0,0,10", "0,0,10,0", "0,0,-5,10", "a,b,c,d", ""])
    def test_malformed_crop_400(self, client, pid, tmp_path, crop):
        r = upload(client, pid, png(bands(800, 600, [RED, BLUE])), crop=crop)
        assert r.status_code == 400 and "detail" in r.json()
        nothing_saved(client, pid, tmp_path)


# ---- rejected bodies ------------------------------------------------------------------------
class TestRejected:
    def test_empty_body_400(self, client, pid, tmp_path):
        r = upload(client, pid, b"")
        assert r.status_code == 400 and "detail" in r.json()
        nothing_saved(client, pid, tmp_path)

    @pytest.mark.parametrize("data", [b"this is not an image", b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, b"GIF89a"])
    def test_not_an_image_415(self, client, pid, tmp_path, data):
        r = upload(client, pid, data)
        assert r.status_code == 415 and "detail" in r.json()
        nothing_saved(client, pid, tmp_path)

    def test_truncated_image_is_not_saved(self, client, pid, tmp_path):
        data = jpeg(bands(1000, 800, [RED, GREEN, BLUE]))
        r = upload(client, pid, data[: len(data) // 3])
        assert r.status_code in (400, 415)
        nothing_saved(client, pid, tmp_path)

    def test_more_than_80_megapixels_413(self, client, pid, tmp_path):
        big = Image.new("1", (9000, 9000))  # 81 MP, a few KB as PNG
        data = png(big)
        assert len(data) < 25 * 1024 * 1024
        r = upload(client, pid, data)
        assert r.status_code == 413 and "detail" in r.json()
        nothing_saved(client, pid, tmp_path)

    def test_just_under_the_pixel_limit_is_accepted(self, client, pid):
        data = png(Image.new("1", (8900, 8900)))  # 79.2 MP
        assert upload(client, pid, data).status_code == 201

    def test_more_than_25mb_413_or_connection_error(self, client, pid, tmp_path):
        body = b"x" * (26 * 1024 * 1024)
        try:
            status = upload(client, pid, body).status_code
        except (httpx.HTTPError, OSError):  # API.md: the server may close the connection while the client is sending
            status = None
        assert status in (None, 413)
        nothing_saved(client, pid, tmp_path)
        # the project is not stuck "in flight": a normal upload still works
        assert upload(client, pid, png(bands(400, 300, [RED, BLUE]))).status_code == 201

    def test_a_valid_image_over_25mb_is_413(self, client, pid, tmp_path):
        noise = Image.frombytes("RGB", (3000, 3000), (bytes(range(256)) * (3000 * 3000 * 3 // 256 + 1))[: 3000 * 3000 * 3])
        data = png(noise)
        data += b"\0" * max(0, 26 * 1024 * 1024 - len(data))  # trailing bytes keep it a valid PNG but over the cap
        try:
            status = upload(client, pid, data).status_code
        except (httpx.HTTPError, OSError):
            status = None
        assert status in (None, 413)
        nothing_saved(client, pid, tmp_path)


# ---- conflicts ------------------------------------------------------------------------------
class TestConflicts:
    def test_unknown_project_404(self, client):
        r = upload(client, "nope", png(bands(100, 100, [RED, BLUE])))
        assert r.status_code == 404

    def test_second_upload_409(self, client, pid):
        assert upload(client, pid, png(bands(400, 300, [RED, BLUE]))).status_code == 201
        r = upload(client, pid, png(bands(400, 300, [GREEN, BLUE])))
        assert r.status_code == 409 and "detail" in r.json()
        assert [x["run"] for x in client.get(f"/api/projects/{pid}").json()["runs"]] == [1]

    def test_project_with_a_generated_run_409(self, client, pid):
        do_run(client, pid, "a cat")
        assert upload(client, pid, png(bands(400, 300, [RED, BLUE]))).status_code == 409

    def test_project_with_a_pending_job_409(self, client, pid, engine, tmp_path):
        engine.gate.clear()
        r = submit(client, pid, "queued")
        assert upload(client, pid, png(bands(400, 300, [RED, BLUE]))).status_code == 409
        nothing_saved(client, pid, tmp_path)
        engine.gate.set()
        wait_run(client, pid, r["run_reserved"])

    def test_other_projects_are_unaffected(self, client, pid, engine):
        engine.gate.clear()
        submit(client, pid, "queued in project 1")
        other = new_project(client)["id"]
        assert upload(client, other, png(bands(400, 300, [RED, BLUE]))).status_code == 201
        engine.gate.set()

    def test_cancelled_job_does_not_block(self, client, pid, engine):
        engine.gate.clear()
        r = submit(client, pid, "doomed")
        client.post(f"/api/jobs/{r['job_id']}/cancel")
        wait_until(lambda: engine.get(r["job_id"])["state"] == "cancelled")
        engine.gate.set()
        assert upload(client, pid, png(bands(400, 300, [RED, BLUE]))).status_code == 201


# ---- after an upload ------------------------------------------------------------------------
class TestAfterUpload:
    def test_next_run_edits_the_upload(self, client, pid, engine):
        upload(client, pid, png(bands(3000, 4000, [RED, BLUE])))
        r = submit(client, pid, "make it green")
        assert r["mode"] == "edit"
        job = engine.submitted[-1]
        assert job["mode"] == "edit" and job["ref_image_path"].endswith("runs/0001.png")
        run = wait_run(client, pid, r["run_reserved"])
        assert run["mode"] == "edit" and run["parent_run"] == 1
        assert run["run"] == 2 and client.get(f"/api/projects/{pid}").json()["current_run"] == 2

    def test_enhance_after_upload_is_i2i_on_the_upload(self, client, pid, engine):
        upload(client, pid, png(bands(3000, 4000, [RED, BLUE])))
        r = client.post(f"/api/projects/{pid}/enhance", json={"prompt": "x"})
        assert r.status_code == 202 and r.json()["mode"] == "i2i"
        assert engine.submitted[-1]["ref_image_path"].endswith("runs/0001.png")

    def test_can_branch_back_to_the_upload(self, client, pid, engine):
        upload(client, pid, png(bands(3000, 4000, [RED, BLUE])))
        do_run(client, pid, "one")
        assert client.patch(f"/api/projects/{pid}", json={"current_run": 1}).status_code == 200
        r = do_run(client, pid, "two")
        assert wait_run(client, pid, r["run_reserved"])["parent_run"] == 1

    def test_run_sse_event_is_broadcast(self, app):
        async def action(c):
            pid = (await c.post("/api/projects", json={})).json()["id"]
            r = await c.post(f"/api/projects/{pid}/upload", content=png(bands(3024, 4032, [RED, BLUE])))
            assert r.status_code == 201
            return pid

        events, pid = collect_events(app, action, lambda ev, _: any(e["event"] == "run" for e in ev))
        runs = [e["data"] for e in events if e["event"] == "run"]
        assert len(runs) == 1 and runs[0]["project_id"] == pid
        run = runs[0]["run"]
        assert run["run"] == 1 and run["mode"] == "upload" and (run["width"], run["height"]) == (896, 1184)
        assert run["image_url"] == f"/api/projects/{pid}/runs/1/image"

    def test_delete_project_removes_the_upload(self, client, pid, tmp_path):
        upload(client, pid, png(bands(400, 300, [RED, BLUE])))
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        assert not (tmp_path / "data" / "projects" / pid).exists()


# ---- resolution must be a multiple of 32 -----------------------------------------------------
class TestResolutionRule:
    @pytest.mark.parametrize("res", [1000, 1025, 1023, 500, 257, 2047])
    def test_non_multiples_are_400(self, client, res):
        before = client.get("/api/settings").json()
        r = client.put("/api/settings", json={"resolution": res})
        assert r.status_code == 400 and "32" in r.json()["detail"]
        assert client.get("/api/settings").json() == before

    @pytest.mark.parametrize("res", [256, 1024, 1056, 2048, 960])
    def test_multiples_are_accepted(self, client, res):
        r = client.put("/api/settings", json={"resolution": res})
        assert r.status_code == 200 and r.json()["resolution"] == res

    @pytest.mark.parametrize("res", [224, 2080, 0, -32])
    def test_range_still_applies(self, client, res):
        assert client.put("/api/settings", json={"resolution": res}).status_code == 400

    @pytest.mark.parametrize("stored_res,want", [(1000, 992), (1010, 1024), (1100, 1088), (1300, 1312)])
    def test_old_settings_file_is_rounded_to_the_nearest_multiple(self, tmp_path, stored_res, want):
        data = tmp_path / "old"
        data.mkdir()
        (data / "settings.json").write_text(json.dumps({"resolution": stored_res, "steps": 8}))
        with TestClient(create_app(engine=StubEngine(), data_dir=data)) as c:
            s = c.get("/api/settings").json()
        assert s["resolution"] == want and s["resolution"] % 32 == 0
        assert s["steps"] == 8  # the rest of the file is kept

    def test_rounded_resolution_feeds_sizes(self, tmp_path):
        data = tmp_path / "old"
        data.mkdir()
        (data / "settings.json").write_text(json.dumps({"resolution": 1000}))
        with TestClient(create_app(engine=StubEngine(), data_dir=data)) as c:
            body = c.get("/api/sizes").json()
        assert body["resolution"] == 992
        assert {p["label"]: (p["width"], p["height"]) for p in body["presets"]}["1:1"] == (992, 992)
