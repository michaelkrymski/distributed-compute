import { getDb } from "../../../db";
import { adminAccounts, adminSessions } from "../../../db/schema";
import { getTeamUser, apiJson } from "../../../lib/access";
import {
  ADMIN_ID,
  adminConfigured,
  adminSession,
  equalHash,
  passwordHash,
  randomSecret,
  sessionCookie,
  sessionToken,
  sha256,
  trustedBrowser,
} from "../../../lib/admin-auth";
import { and, eq, lt, sql } from "drizzle-orm";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  return apiJson({
    configured: await adminConfigured(),
    authenticated: await adminSession(request),
  });
}
export async function authenticate(request: Request, desktop = false) {
  if (!desktop && !trustedBrowser(request))
    return apiJson({ error: "Use the dashboard to sign in." }, 403);
  if (!request.headers.get("content-type")?.includes("application/json"))
    return apiJson({ error: "JSON required." }, 415);
  const raw = await request.text();
  if (raw.length > 4096) return apiJson({ error: "Request too large." }, 413);
  let b;
  try {
    b = JSON.parse(raw);
  } catch {
    return apiJson({ error: "Invalid JSON." }, 400);
  }
  if (!b || typeof b !== "object")
    return apiJson({ error: "Invalid request." }, 400);
  const db = getDb(),
    now = Date.now();
  if (desktop && b.action !== "login") return apiJson({error:"Login required"},400);
  if (b.action === "logout") {
    await db
      .delete(adminSessions)
      .where(eq(adminSessions.tokenHash, await sha256(sessionToken(request))));
    const response = apiJson({ ok: true });
    response.headers.set("Set-Cookie", sessionCookie(request, "", 0));
    return response;
  }
  if (b.action !== "setup" && b.action !== "login")
    return apiJson({ error: "Unknown action." }, 400);
  if (
    typeof b.password !== "string" ||
    b.password.length < 16 ||
    b.password.length > 256
  )
    return apiJson(
      { error: "Use a password between 16 and 256 characters." },
      400,
    );
  if (b.action === "setup") {
    // The existing coordinator node key activates the browser account once.
    const user = await getTeamUser(request);
    if (!user || user.role !== "coordinator" || user.userId !== ADMIN_ID)
      return apiJson(
        { error: "Your coordinator key is required for initial setup." },
        403,
      );
    if (await adminConfigured())
      return apiJson(
        { error: "Admin password already set. Sign in instead." },
        409,
      );
    const salt = randomSecret();
    const inserted = await db
      .insert(adminAccounts)
      .values({
        id: ADMIN_ID,
        salt,
        passwordHash: await passwordHash(b.password, salt),
        failures: 0,
        lockedUntil: 0,
      })
      .onConflictDoNothing()
      .returning({ id: adminAccounts.id });
    if (!inserted.length)
      return apiJson({ error: "Admin password already set." }, 409);
  } else {
    const account = (
      await db
        .select()
        .from(adminAccounts)
        .where(eq(adminAccounts.id, ADMIN_ID))
    )[0];
    if (!account)
      return apiJson({ error: "Set up the admin account first." }, 401);
    // Reserve attempts atomically: at most five password checks per 15 minutes,
    // including concurrent requests. A successful sign-in resets the counter.
    await db
      .update(adminAccounts)
      .set({ failures: 0, lockedUntil: 0 })
      .where(
        and(
          eq(adminAccounts.id, ADMIN_ID),
          lt(adminAccounts.lockedUntil, now),
          sql`${adminAccounts.lockedUntil} != 0`,
        ),
      );
    const attempt = await db
      .update(adminAccounts)
      .set({
        failures: sql`${adminAccounts.failures} + 1`,
        lockedUntil: sql`CASE WHEN ${adminAccounts.lockedUntil}=0 THEN ${now + 900000} ELSE ${adminAccounts.lockedUntil} END`,
      })
      .where(and(eq(adminAccounts.id, ADMIN_ID), lt(adminAccounts.failures, 5)))
      .returning({ id: adminAccounts.id });
    if (!attempt.length)
      return apiJson(
        { error: "Too many attempts. Try again after 15 minutes." },
        429,
      );
    if (
      !equalHash(
        await passwordHash(b.password, account.salt),
        account.passwordHash,
      )
    )
      return apiJson({ error: "Incorrect password." }, 401);
    await db
      .update(adminAccounts)
      .set({ failures: 0, lockedUntil: 0 })
      .where(eq(adminAccounts.id, ADMIN_ID));
  }
  const token = randomSecret();
  await db.delete(adminSessions).where(lt(adminSessions.expiresAt, now));
  await db
    .insert(adminSessions)
    .values({ tokenHash: await sha256(token), expiresAt: now + 43200000 });
  if (desktop) {
    await db.run(sql`INSERT INTO desktop_sessions(token_hash,user_id,expires_at) VALUES(${await sha256(token)},${ADMIN_ID},${now + 2592000000})`);
    return apiJson({token, expiresAt: now + 2592000000});
  }
  const response = apiJson({ ok: true });
  response.headers.set("Set-Cookie", sessionCookie(request, token));
  return response;
}

export async function POST(request: Request) { return authenticate(request); }
