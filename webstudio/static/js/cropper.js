// Crop editor for "start a project from an image": fixed-aspect frame over the photo, pan/zoom but never zoom out past
// cover (no padding), then render the crop at the preset's exact output size and upload it.
import { h, fill, toast } from "./util.js";
import { icon } from "./icons.js";
import { zoomPan } from "./gestures.js";

const MAX_SCALE = 8; // relative to cover

async function loadBitmap(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" }); // EXIF orientation applied
  } catch {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode(); // browsers orient <img> by EXIF
      return img;
    } finally {
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }
}
const dimsOf = (b) => [b.width ?? b.naturalWidth, b.height ?? b.naturalHeight];

// Draw the source region (sx,sy,sw,sh) of `img` into a dw x dh canvas. Large reductions are done in halving steps so
// browsers that only do bilinear sampling (mobile Safari) don't alias. The canvas is filled opaque first.
function renderCrop(img, sx, sy, sw, sh, dw, dh) {
  const mk = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });
  let src = img, rx = sx, ry = sy, rw = sw, rh = sh;
  while (rw / 2 >= dw && rh / 2 >= dh) {
    const nw = Math.ceil(rw / 2), nh = Math.ceil(rh / 2);
    const c = mk(nw, nh), g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(src, rx, ry, rw, rh, 0, 0, nw, nh);
    src = c; rx = ry = 0; rw = nw; rh = nh;
  }
  const out = mk(dw, dh), g = out.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, dw, dh);
  g.imageSmoothingQuality = "high";
  g.drawImage(src, rx, ry, rw, rh, 0, 0, dw, dh);
  return out;
}

// Opens the editor. opts: {pid, api, onUploaded(run), onCancel()}. Resolves when closed.
export async function openCropper(file, { pid, api, onUploaded, onCancel }) {
  let bitmap;
  try {
    bitmap = await loadBitmap(file);
  } catch {
    toast("Couldn't read that image. Try a JPEG or PNG.", "error");
    onCancel?.();
    return;
  }
  const [SW, SH] = dimsOf(bitmap);
  let info;
  try {
    info = await api("GET", `/sizes?w=${SW}&h=${SH}`);
  } catch (e) {
    toast(e.message, "error");
    onCancel?.();
    return;
  }
  let preset = info.presets.find((p) => p.label === info.closest) ?? info.presets[0];

  // A downscaled copy for display only (the full-size bitmap is used for the final render).
  const prevScale = Math.min(1, 2048 / Math.max(SW, SH));
  const view = Object.assign(document.createElement("canvas"), { width: Math.round(SW * prevScale), height: Math.round(SH * prevScale) });
  view.className = "crop-img";
  const vg = view.getContext("2d");
  vg.imageSmoothingQuality = "high";
  vg.drawImage(bitmap, 0, 0, view.width, view.height);

  const frame = h("div", { class: "crop-frame" }, h("i", {}), h("i", {}), h("i", {}), h("i", {}));
  const stage = h("div", { class: "crop-stage" }, view, frame);
  const readout = h("div", { class: "crop-readout" });
  const err = h("div", { class: "field-err", hidden: true, role: "alert" });
  const chips = h("div", { class: "crop-chips", role: "radiogroup", "aria-label": "Aspect ratio" });
  const useBtn = h("button", { class: "btn primary", type: "button", onclick: () => use() }, "Use image");
  const cancelBtn = h("button", { class: "btn", type: "button", onclick: () => close(true) }, "Cancel");
  const root = h("div", { class: "backdrop sheet-wrap crop-wrap" },
    h("div", { class: "cropper", role: "dialog", "aria-label": "Crop image" },
      h("div", { class: "sheet-head" }, h("h3", {}, "Crop image"), h("div", { class: "grow" }),
        h("button", { class: "icon-btn", "aria-label": "Cancel", onclick: () => close(true) }, icon("close"))),
      stage,
      h("div", { class: "crop-foot" }, chips, readout, err, h("div", { class: "row end" }, cancelBtn, useBtn))));

  // Layout: frame fits the stage at the preset ratio; the base image covers the frame exactly.
  let fw = 0, fh = 0, bw = 0, bh = 0, cover = 1, scale = 1, pan = { x: 0, y: 0 };
  function layout() {
    const margin = 18;
    const aw = Math.max(80, stage.clientWidth - margin * 2), ah = Math.max(80, stage.clientHeight - margin * 2);
    const r = preset.width / preset.height;
    fw = Math.min(aw, ah * r);
    fh = fw / r;
    cover = Math.max(fw / SW, fh / SH);
    bw = SW * cover;
    bh = SH * cover;
    Object.assign(frame.style, { width: `${fw}px`, height: `${fh}px`, marginLeft: `${-fw / 2}px`, marginTop: `${-fh / 2}px` });
    Object.assign(view.style, { width: `${bw}px`, height: `${bh}px`, marginLeft: `${-bw / 2}px`, marginTop: `${-bh / 2}px` });
  }
  // The crop box in ORIGINAL oriented source pixels, derived from the current scale and pan.
  function cropBox() {
    const k = cover * scale;
    const w = fw / k, hh = fh / k;
    const cx = SW / 2 - pan.x / k, cy = SH / 2 - pan.y / k;
    const x = Math.min(Math.max(0, cx - w / 2), SW - w), y = Math.min(Math.max(0, cy - hh / 2), SH - hh);
    return { x, y, w, h: hh };
  }
  function readoutUpdate() {
    const b = cropBox();
    const keep = Math.round((100 * b.w * b.h) / (SW * SH));
    readout.textContent = `${preset.width}×${preset.height} · ${preset.label} · keeps ${keep}% of the photo`;
  }
  const zp = zoomPan(stage, {
    limits: () => ({ imgW: bw, imgH: bh, boxW: fw, boxH: fh }),
    maxScale: MAX_SCALE,
    doubleTapScale: null, // double tap/click resets to the default cover crop
    onChange: ({ s, x, y }, animate) => {
      scale = s; pan = { x, y };
      view.classList.toggle("animate", animate);
      view.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
      readoutUpdate();
    },
  });
  function choose(p) {
    preset = p;
    for (const c of chips.children) c.setAttribute("aria-checked", c.dataset.label === p.label);
    layout();
    zp.reset();
  }
  for (const p of info.presets) {
    chips.append(h("button", { class: "chip", type: "button", role: "radio", dataset: { label: p.label }, onclick: () => busy || choose(p) }, p.label));
  }

  let busy = false;
  function setBusy(v) {
    busy = v;
    useBtn.disabled = cancelBtn.disabled = v;
    useBtn.textContent = v ? "Uploading…" : "Use image";
    chips.classList.toggle("disabled", v);
  }
  const showErr = (m) => {
    err.hidden = !m;
    err.textContent = m || "";
  };
  async function use() {
    if (busy) return;
    showErr("");
    setBusy(true);
    try {
      const b = cropBox();
      const sx = Math.round(b.x), sy = Math.round(b.y);
      const sw = Math.min(SW - sx, Math.round(b.w)), sh = Math.min(SH - sy, Math.round(b.h));
      const canvas = renderCrop(bitmap, sx, sy, sw, sh, preset.width, preset.height);
      const blob = await new Promise((res, rej) => canvas.toBlob((bl) => (bl ? res(bl) : rej(new Error("render failed"))), "image/png"));
      const q = new URLSearchParams({ preset: preset.label, crop: [sx, sy, sw, sh].join(","), src: `${SW}x${SH}` });
      let res;
      try {
        res = await fetch(`/api/projects/${pid}/upload?${q}`, { method: "POST", body: blob });
      } catch {
        throw new Error("Couldn't reach the server. Check your connection and try again.");
      }
      if (!res.ok) {
        let d = "";
        try { d = (await res.json()).detail; } catch {}
        const msg = { 409: "This project already has an image.", 413: "That image is too large to upload.", 415: "The server couldn't read that image." }[res.status];
        throw new Error(msg || d || `Upload failed (${res.status})`);
      }
      const run = await res.json();
      close(false);
      onUploaded?.(run);
    } catch (e) {
      setBusy(false);
      showErr(e.message);
    }
  }
  function close(cancelled) {
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("resize", onResize);
    root.classList.remove("in");
    document.body.classList.remove("no-scroll");
    setTimeout(() => root.remove(), 200);
    bitmap.close?.();
    if (cancelled) onCancel?.();
  }
  function onKey(e) {
    if (e.key === "Escape") {
      e.stopPropagation();
      if (!busy) close(true);
    }
  }
  const onResize = () => {
    layout();
    zp.reset();
  };

  document.body.append(root);
  document.body.classList.add("no-scroll");
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", onResize);
  requestAnimationFrame(() => {
    root.classList.add("in");
    choose(preset);
  });
}
