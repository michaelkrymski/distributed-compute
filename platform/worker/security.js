const enc = new TextEncoder();
const hex = value => Array.from(new Uint8Array(value), x => x.toString(16).padStart(2, "0")).join("");
export const secret = (prefix = "") => prefix + hex(crypto.getRandomValues(new Uint8Array(32)));
export const digest = async value => hex(await crypto.subtle.digest("SHA-256", enc.encode(value)));
export function fail(message, status = 400) { throw Object.assign(new Error(message), { status }); }
export function text(value, name, max = 100) {
  if (typeof value !== "string" || !value.trim() || value.length > max) fail(`Invalid ${name}.`);
  return value.trim();
}
export function username(value) {
  const name = text(value, "username", 40).toLowerCase();
  if (!/^[a-z0-9][a-z0-9_.-]{2,39}$/.test(name)) fail("Use 3–40 letters, digits, dots, underscores or hyphens for your username.");
  return name;
}
export function password(value) {
  if (typeof value !== "string" || value.length < 16 || value.length > 256) fail("Use a password of 16–256 characters.");
  return value;
}
export async function passwordHash(value, salt) {
  const key = await crypto.subtle.importKey("raw", enc.encode(value), "PBKDF2", false, ["deriveBits"]);
  return hex(await crypto.subtle.deriveBits({ name: "PBKDF2", salt: enc.encode(salt), iterations: 100000, hash: "SHA-256" }, key, 256));
}
export function equal(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let difference = 0; for (let i = 0; i < a.length; i++) difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return difference === 0;
}
export async function rate(db, scope, identity, limit, seconds) {
  const now = Date.now(), bucket = Math.floor(now / (seconds * 1000));
  const id = `${scope}:${await digest(identity)}:${bucket}`;
  const result = await db.prepare("INSERT INTO rate_limits(id,hits,expires_at) VALUES(?,1,?) ON CONFLICT(id) DO UPDATE SET hits=hits+1 WHERE hits<? RETURNING hits").bind(id, now + seconds * 2000, limit).first();
  if (!result) fail("Too many attempts. Please wait before trying again.", 429);
}
export async function body(request) {
  if (!request.headers.get("content-type")?.includes("application/json")) fail("JSON required.", 415);
  if (Number(request.headers.get("content-length") || 0) > 65536) fail("Request too large.", 413);
  const raw = await request.text(); if (raw.length > 65536) fail("Request too large.", 413);
  let value; try { value = JSON.parse(raw); } catch { fail("Invalid JSON."); }
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("Invalid request.");
  return value;
}
export function bearer(request) {
  const h = request.headers.get("authorization") || "";
  if (!/^Bearer [a-zA-Z0-9_]{40,120}$/.test(h)) fail("Sign in or provide a valid access key.", 401);
  return h.slice(7);
}
export async function account(request, db) {
  const token = bearer(request);
  const row = await db.prepare("SELECT a.id,a.username FROM platform_sessions s JOIN platform_accounts a ON a.id=s.account_id WHERE s.token_hash=? AND s.expires_at>?").bind(await digest(token), Date.now()).first();
  if (!row) fail("Your session expired. Sign in again.", 401);
  return row;
}
export async function member(db, teamId, accountId, admin = false) {
  const row = await db.prepare("SELECT m.*,a.username FROM team_members m JOIN platform_accounts a ON a.id=m.account_id WHERE m.team_id=? AND m.account_id=?").bind(teamId, accountId).first();
  if (!row) fail("Team not found or access denied.", 404);
  if (admin && row.role !== "admin") fail("Team administrator access required.", 403);
  return row;
}
export async function workerIdentity(request, db) {
  const token = bearer(request);
  const row = await db.prepare("SELECT k.id AS key_id,k.team_id,k.project_id,k.account_id,m.role,a.username FROM worker_keys k JOIN team_members m ON m.team_id=k.team_id AND m.account_id=k.account_id JOIN platform_accounts a ON a.id=k.account_id JOIN compute_projects p ON p.id=k.project_id AND p.team_id=k.team_id WHERE k.token_hash=? AND k.revoked_at IS NULL").bind(await digest(token)).first();
  if (!row) fail("Access key revoked or invalid.", 401);
  return { ...row, userId: row.username, role: row.role === "admin" ? "coordinator" : "member" };
}
export async function audit(db, teamId, actor, action, target = "") {
  await db.prepare("INSERT INTO audit_events(id,team_id,actor_id,action,target,created_at) VALUES(?,?,?,?,?,?)").bind(crypto.randomUUID(), teamId, actor, action, target, Date.now()).run();
}
