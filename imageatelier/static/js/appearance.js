// Pictures and favicon media queries switch natively. Keep install metadata in sync too.
const appearance = window.matchMedia("(prefers-color-scheme: dark)");
function syncAppearance() {
  const mode = appearance.matches ? "dark" : "light";
  document.querySelector('link[rel="manifest"]').href = mode === "dark"
    ? "/manifest-dark.webmanifest" : "/manifest.webmanifest";
  document.querySelector('link[rel="apple-touch-icon"]').href = `/icons/${mode}/apple-touch-icon.png`;
}
appearance.addEventListener("change", syncAppearance);
syncAppearance();
