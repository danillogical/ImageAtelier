// Image mode options shared by the composer dropdown and the Settings sheet, so they can never disagree.
// A generation = an image model + a text encoder (chosen in Settings) + a mode (Turbo or Standard, chosen here).
import { S } from "./state.js";

export const imageModels = () => S.models.filter((m) => m.kind === "image");
export const textEncoders = () => S.models.filter((m) => m.kind === "text_encoder");
export const currentImage = () => imageModels().find((m) => m.name === S.settings?.image_model) ?? null;
export const currentEncoder = () => textEncoders().find((m) => m.name === S.settings?.text_encoder) ?? null;
export const encoderName = (n) => (String(n).startsWith("heretic") ? "Heretic" : "Official");
export const prettyLabel = (label) => String(label).replace("Qwen-Image-2.1", "Qwen-Image 2.1").replace(/, 8-bit$/, " · 8-bit");

// Modes available for the selected image model: Turbo needs the turbo add-on. Order: Turbo, Standard.
export function modeOptions() {
  const m = currentImage();
  if (m?.state !== "ready") return [];
  return m.turbo_lora_ready ? [{ turbo: true }, { turbo: false }] : [{ turbo: false }];
}
export const optName = (o) => (o.turbo ? "Turbo" : "Standard");
export const isCurrent = (o) => !!S.settings && o.turbo === S.settings.turbo;
// Steps the server will use after switching: it keeps them unless turbo changes, then resets (6 turbo / 40).
export const stepsAfter = (o) => (o.turbo === S.settings.turbo ? S.settings.steps : o.turbo ? 6 : 40);
export const tagline = (o) => `${o.turbo ? "Fast" : "Best quality"} · ${stepsAfter(o)} steps`;
