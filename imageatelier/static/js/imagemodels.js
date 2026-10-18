// Display helpers for current and historical run metadata.
export const encoderName = (name) => String(name).startsWith("custom-") ? `Custom Qwen (${String(name).split("@")[0].slice(7, 15)})` : `Qwen (${String(name).replace("official-", "")})`;
