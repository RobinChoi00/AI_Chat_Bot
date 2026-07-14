import { createHmac, timingSafeEqual } from "crypto";

export const ADMIN_COOKIE_NAME = "admin_session";
export const ADMIN_SESSION_MAX_AGE_SECONDS = 60 * 60 * 8;

type AdminSessionPayload = {
  sub: string;
  exp: number;
};

function configuredAdmin(): { username: string; secret: string } | null {
  const username = process.env.ADMIN_USERNAME?.trim();
  const secret = process.env.ADMIN_SESSION_SECRET?.trim();
  if (!username || !secret || secret.length < 32) return null;
  return { username, secret };
}

function signature(payload: string, secret: string): string {
  return createHmac("sha256", secret).update(payload).digest("hex");
}

function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

export function createAdminSessionToken(nowSeconds = Math.floor(Date.now() / 1000)): string {
  const config = configuredAdmin();
  if (!config) throw new Error("Admin session configuration is missing or too weak.");

  const payload: AdminSessionPayload = {
    sub: config.username,
    exp: nowSeconds + ADMIN_SESSION_MAX_AGE_SECONDS,
  };
  const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
  return `${encoded}.${signature(encoded, config.secret)}`;
}

export function verifyAdminSessionToken(token: string, nowSeconds = Math.floor(Date.now() / 1000)): boolean {
  const config = configuredAdmin();
  if (!config) return false;

  const [encoded, suppliedSignature, extra] = token.split(".");
  if (!encoded || !suppliedSignature || extra) return false;
  if (!safeEqual(suppliedSignature, signature(encoded, config.secret))) return false;

  try {
    const payload = JSON.parse(
      Buffer.from(encoded, "base64url").toString("utf8")
    ) as Partial<AdminSessionPayload>;
    return (
      payload.sub === config.username &&
      typeof payload.exp === "number" &&
      Number.isFinite(payload.exp) &&
      payload.exp > nowSeconds
    );
  } catch {
    return false;
  }
}

function cookieValue(request: Request, name: string): string {
  const cookieHeader = request.headers.get("cookie") ?? "";
  for (const part of cookieHeader.split(";")) {
    const [rawName, ...rawValue] = part.trim().split("=");
    if (rawName === name) return decodeURIComponent(rawValue.join("="));
  }
  return "";
}

export function isAdminApiRequestAuthenticated(request: Request): boolean {
  return verifyAdminSessionToken(cookieValue(request, ADMIN_COOKIE_NAME));
}
