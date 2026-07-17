/**
 * Resolve which Shopify store domain to send to warranty API calls.
 *
 * Priority:
 *  1. ?store= query param (set by warranty-launcher.js from parent hostname)
 *  2. document.referrer host (Shopify parent page when embedded)
 *  3. current iframe hostname
 */
export function resolveWarrantyStoreDomain(): string {
  if (typeof window === "undefined") {
    return "osakiusa.com";
  }

  const fromQuery = new URLSearchParams(window.location.search).get("store")?.trim();
  if (fromQuery) {
    return fromQuery.replace(/^https?:\/\//i, "").replace(/\/$/, "").toLowerCase();
  }

  try {
    if (document.referrer) {
      const refHost = new URL(document.referrer).hostname.toLowerCase();
      const selfHost = window.location.hostname.toLowerCase();
      if (refHost && refHost !== selfHost) {
        return refHost;
      }
    }
  } catch {
    /* ignore malformed referrer */
  }

  return (window.location.hostname || "osakiusa.com").toLowerCase();
}
