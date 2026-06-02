/**
 * Backend base URL for server-side fetches (Route Handlers, Server Components).
 *
 * In Docker Compose, prefer BACKEND_INTERNAL_URL (http://backend:8000) so the
 * Next.js container talks to FastAPI on the internal network instead of looping
 * out through the public API domain (which often hangs or times out on EC2).
 *
 * Browser/client code should keep using NEXT_PUBLIC_API_BASE_URL via lib/api.ts.
 */
export function getBackendUrl(): string {
  const internal = process.env.BACKEND_INTERNAL_URL?.trim();
  if (internal) {
    return internal.replace(/\/$/, "");
  }
  return (
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
  ).replace(/\/$/, "");
}
