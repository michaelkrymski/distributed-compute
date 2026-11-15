import { getDb } from "../db";
import { adminAccounts, adminSessions } from "../db/schema";
import { and, eq, gt } from "drizzle-orm";

export const ADMIN_ID = "michaelkrymski";
const enc = new TextEncoder();
const hex = (bytes: ArrayBuffer | Uint8Array) =>
  Array.from(new Uint8Array(bytes))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
export const randomSecret = () =>
  hex(crypto.getRandomValues(new Uint8Array(32)));
export async function sha256(value: string) {
  return hex(await crypto.subtle.digest("SHA-256", enc.encode(value)));
}
export async function passwordHash(password: string, salt: string) {
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(password),
    "PBKDF2",
    false,
    ["deriveBits"],
  );
  // workerd caps PBKDF2 at 100,000 iterations. Never store the password.
  return hex(
    await crypto.subtle.deriveBits(
      {
        name: "PBKDF2",
        salt: enc.encode(salt),
        iterations: 100_000,
        hash: "SHA-256",
      },
      key,
      256,
    ),
  );
}
export function equalHash(a: string, b: string) {
  if (a.length !== b.length) return false;
  let difference = 0;
  for (let i = 0; i < a.length; i++)
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return difference === 0;
}
export function trustedBrowser(request: Request) {
  const origin = request.headers.get("origin");
  return (
    origin === "https://frontend.example.invalid" ||
    origin === "https://backend.example.invalid" ||
    (new URL(request.url).hostname === "127.0.0.1" &&
      /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(origin ?? ""))
  );
}
function cookieName(request: Request) {
  return new URL(request.url).protocol === "https:"
    ? "__Host-gator-admin"
    : "gator-admin-local";
}
export function sessionToken(request: Request) {
  return (
    request.headers
      .get("cookie")
      ?.split(";")
      .map((v) => v.trim())
      .find((v) => v.startsWith(cookieName(request) + "="))
      ?.split("=")[1] ?? ""
  );
}
export function sessionCookie(request: Request, token: string, age = 43200) {
  return `${cookieName(request)}=${token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${age}${new URL(request.url).protocol === "https:" ? "; Secure" : ""}`;
}
export async function adminSession(request: Request) {
  const token = sessionToken(request);
  if (!/^[a-f0-9]{64}$/.test(token)) return false;
  if (request.method !== "GET" && !trustedBrowser(request)) return false;
  const rows = await getDb()
    .select()
    .from(adminSessions)
    .where(
      and(
        eq(adminSessions.tokenHash, await sha256(token)),
        gt(adminSessions.expiresAt, Date.now()),
      ),
    );
  return rows.length === 1;
}
export async function adminConfigured() {
  return (
    (
      await getDb()
        .select({ id: adminAccounts.id })
        .from(adminAccounts)
        .where(eq(adminAccounts.id, ADMIN_ID))
    ).length > 0
  );
}
