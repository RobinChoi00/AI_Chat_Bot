import { NextResponse } from "next/server";
import { isAdminApiRequestAuthenticated } from "@/lib/adminSession";
import { getBackendUrl } from "@/lib/backendUrl";

/** Fixed upstream paths only. Never send the backend admin key to the browser. */
export async function proxyKnowledgeWrite(request: Request, reviewId?: string) {
  let authenticated = false;
  try { authenticated = isAdminApiRequestAuthenticated(request); } catch { /* invalid cookie */ }
  if (!authenticated) return NextResponse.json({ detail: "Admin authentication required." }, { status: 401 });
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin) {
    return NextResponse.json({ detail: "Same-origin request required." }, { status: 403 });
  }
  const key = process.env.ADMIN_API_KEY;
  if (!key) return NextResponse.json({ detail: "ADMIN_API_KEY is not configured." }, { status: 503 });
  if (reviewId !== undefined && !/^[1-9]\d*$/.test(reviewId)) {
    return NextResponse.json({ detail: "Invalid review ID." }, { status: 400 });
  }
  try {
    const text = await request.text();
    if (text.length > 32_000) return NextResponse.json({ detail: "Draft is too large." }, { status: 413 });
    const body = JSON.parse(text) as Record<string, unknown>;
    if (!body || typeof body !== "object" || Array.isArray(body)) {
      return NextResponse.json({ detail: "JSON object required." }, { status: 400 });
    }
    if (reviewId) body.reviewed_by = process.env.ADMIN_USERNAME ?? "admin";
    const path = `/admin/warranty/knowledge-reviews${reviewId ? `/${reviewId}/decision` : ""}`;
    const response = await fetch(`${getBackendUrl()}${path}`, {
      method: "POST",
      headers: { "X-Admin-Key": key, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof SyntaxError ? "Invalid JSON body." : "Knowledge service is unavailable." },
      { status: error instanceof SyntaxError ? 400 : 502 }
    );
  }
}
