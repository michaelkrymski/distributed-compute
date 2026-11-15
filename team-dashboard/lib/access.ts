import { env } from "cloudflare:workers";
import { adminSession, ADMIN_ID } from "./admin-auth";
export type TeamUser = {
  userId: string;
  displayName: string;
  role: "member" | "coordinator";
};
export async function getTeamUser(request: Request): Promise<TeamUser | null> {
  const header = request.headers.get("Authorization") ?? "";
  if (!header && (await adminSession(request)))
    return { userId: ADMIN_ID, displayName: ADMIN_ID, role: "coordinator" };
  if (!header.startsWith("Bearer ") || header.length > 256) return null;
  const digest = Array.from(
    new Uint8Array(
      await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(header.slice(7)),
      ),
    ),
  )
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  const session = env.DB ? await env.DB.prepare("SELECT user_id FROM desktop_sessions WHERE token_hash=? AND expires_at>?").bind(digest,Date.now()).first() : null;
  if(session && session.user_id === ADMIN_ID) return {userId: ADMIN_ID, displayName: ADMIN_ID,role:"coordinator"};
  let keys: Record<string, { sha256: string; role: "member" | "coordinator" }>;
  try {
    keys = JSON.parse(env.TEAM_KEYS_JSON ?? "{}");
  } catch {
    return null;
  }
  for (const [id, entry] of Object.entries(keys)) {
    if (
      !entry ||
      typeof entry.sha256 !== "string" ||
      entry.sha256.length !== digest.length
    )
      continue;
    let different = 0;
    for (let i = 0; i < digest.length; i++)
      different |= digest.charCodeAt(i) ^ entry.sha256.charCodeAt(i);
    if (different === 0)
      return {
        userId: id,
        displayName: id,
        role: entry.role === "coordinator" ? "coordinator" : "member",
      };
  }
  return null;
}
export function apiJson(value: unknown, status = 200) {
  return Response.json(value, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Authorization, Content-Type",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    },
  });
}
export function apiOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Authorization, Content-Type",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Max-Age": "600",
    },
  });
}
export const roster = [
  "onlyspxctre",
  "michaelkrymski",
  "jsaluja09",
  "jacklowry08",
  "carlimarquilage-ctrl",
];
