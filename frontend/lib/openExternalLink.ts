/** Open a URL in a new tab; when embedded, ask the parent Shopify page to open it. */
export function openExternalLink(url: string, event?: { preventDefault?: () => void }) {
  event?.preventDefault?.();

  if (typeof window === "undefined") return;

  if (window.parent && window.parent !== window) {
    window.parent.postMessage({ type: "osaki-warranty-open-link", url }, "*");
  }

  window.open(url, "_blank", "noopener,noreferrer");
}
