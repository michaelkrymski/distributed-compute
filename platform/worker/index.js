import { account, member, workerIdentity, audit, bearer, body, digest, equal, fail, password, passwordHash, rate, secret, text, username } from "./security.js";
import { projectSpec } from "./model.js";
import { coordinate, stateRead } from "./coordination.js";

const FRONTEND = "https://michaelkrymski.me";
const SAFE_KEY_FIELDS = "id,team_id,project_id,account_id,prefix,label,generation,created_at,revoked_at";
const json = (value, status = 200) => Response.json(value, { status, headers: {"cache-control":"no-store", "x-content-type-options":"nosniff"} });
const all = async (db, sql, ...values) => (await db.prepare(sql).bind(...values).all()).results;

async function newTeam(db, owner, b) {
  const now = Date.now(), teamId = crypto.randomUUID(), projectId = crypto.randomUUID();
  const name = text(b.teamName, "team name"), spec = projectSpec(b.project || {});
  const membershipCount = await db.prepare("SELECT count(*) AS n FROM team_members WHERE account_id=? AND role='admin'").bind(owner).first();
  if (membershipCount.n >= 20) fail("This account has reached its team limit.", 409);
  return { teamId, projectId, statements: [
    db.prepare("INSERT INTO compute_teams(id,name,created_at) VALUES(?,?,?)").bind(teamId, name, now),
    db.prepare("INSERT INTO team_members(team_id,account_id,role) VALUES(?,?,'admin')").bind(teamId, owner),
    db.prepare("INSERT INTO compute_projects(id,team_id,spec,revision,created_at) VALUES(?,?,?,1,?)").bind(projectId, teamId, JSON.stringify(spec), now),
  ] };
}
async function project(db, teamId, id) {
  const p = await db.prepare("SELECT * FROM compute_projects WHERE id=? AND team_id=?").bind(id, teamId).first();
  if (!p) fail("Project not found.", 404);
  return { ...p, spec: JSON.parse(p.spec) };
}
async function issueSession(db, id) {
  const token = secret("dc_session_"), expiresAt = Date.now() + 12 * 3600000;
  await db.prepare("INSERT INTO platform_sessions(token_hash,account_id,expires_at) VALUES(?,?,?)").bind(await digest(token), id, expiresAt).run();
  await db.batch([db.prepare("DELETE FROM platform_sessions WHERE expires_at<?").bind(Date.now()),db.prepare("DELETE FROM rate_limits WHERE expires_at<?").bind(Date.now())]);
  return { token, expiresAt };
}

async function platform(request, env, parts) {
  const db = env.DB, now = Date.now(), action = parts[0];
  const ip = request.headers.get("CF-Connecting-IP") || "local";
  if (["setup", "signup", "login"].includes(action) && request.method === "POST") {
    const b = await body(request), name = username(b.username), pass = password(b.password);
    await rate(db, "auth-ip", ip, 40, 900);
    if (action === "login") {
      await rate(db, "login-user", name, 8, 900);
      const row = await db.prepare("SELECT * FROM platform_accounts WHERE username=?").bind(name).first();
      const computed = await passwordHash(pass, row?.salt || "unregistered-user-constant-salt");
      if (!row || !equal(computed, row.password_hash)) fail("Incorrect username or password.", 401);
      return json({ user: { id: row.id, username: row.username }, ...await issueSession(db, row.id) });
    }
    await rate(db, "signup-ip", ip, 12, 3600);
    const id = crypto.randomUUID(), salt = secret(), hash = await passwordHash(pass, salt);
    const insert = db.prepare("INSERT INTO platform_accounts(id,username,salt,password_hash,created_at) VALUES(?,?,?,?,?)").bind(id, name, salt, hash, now);
    const team = action === "setup" ? await newTeam(db, id, b) : null;
    try { await db.batch([insert, ...(team?.statements || [])]); }
    catch (error) { if (String(error).includes("UNIQUE")) fail("That username is already registered. Sign in instead.", 409); throw error; }
    return json({ user: {id, username:name}, teamId:team?.teamId, projectId:team?.projectId, ...await issueSession(db, id) }, 201);
  }

  const user = await account(request, db);
  if (action === "logout" && request.method === "POST") {
    await db.prepare("DELETE FROM platform_sessions WHERE token_hash=?").bind(await digest(bearer(request))).run(); return json({ok:true});
  }
  if (action === "me" && request.method === "GET") {
    return json({ user, teams: await all(db, "SELECT t.id,t.name,m.role FROM compute_teams t JOIN team_members m ON m.team_id=t.id WHERE m.account_id=? ORDER BY t.created_at", user.id) });
  }
  if (action === "join" && request.method === "POST") {
    const b = await body(request), hash = await digest(text(b.invite, "invitation", 120));
    const invitation = await db.prepare("SELECT * FROM team_invitations WHERE token_hash=? AND claimed_by IS NULL AND expires_at>?").bind(hash, now).first();
    if (!invitation) fail("Invitation expired or already used.", 404);
    await db.batch([
      db.prepare("UPDATE team_invitations SET claimed_by=? WHERE id=? AND claimed_by IS NULL AND expires_at>?").bind(user.id, invitation.id, now),
      db.prepare("INSERT INTO team_members(team_id,account_id,role) SELECT team_id,?,'member' FROM team_invitations WHERE id=? AND claimed_by=? ON CONFLICT DO NOTHING").bind(user.id, invitation.id, user.id),
    ]);
    await member(db, invitation.team_id, user.id);
    return json({ok:true,teamId:invitation.team_id});
  }
  if (action !== "teams") fail("Endpoint not found.", 404);
  if (!parts[1] && request.method === "POST") {
    await rate(db, "create-team", user.id, 5, 86400);
    const team = await newTeam(db, user.id, await body(request));
    await db.batch(team.statements); return json({teamId:team.teamId,projectId:team.projectId}, 201);
  }
  const teamId = parts[1], section = parts[2];
  const membership = await member(db, teamId, user.id);
  const admin = membership.role === "admin";
  if (!section && request.method === "GET") {
    const team = await db.prepare("SELECT * FROM compute_teams WHERE id=?").bind(teamId).first();
    return json({ team, role:membership.role,
      members: await all(db, "SELECT m.account_id,m.role,a.username FROM team_members m JOIN platform_accounts a ON a.id=m.account_id WHERE m.team_id=? ORDER BY a.username", teamId),
      projects: (await all(db, "SELECT * FROM compute_projects WHERE team_id=? ORDER BY created_at", teamId)).map(p => ({...p,spec:JSON.parse(p.spec)})),
      keys: await all(db, `SELECT ${SAFE_KEY_FIELDS} FROM worker_keys WHERE team_id=?${admin ? "" : " AND account_id=?"} ORDER BY created_at DESC`, ...[teamId, ...(admin ? [] : [user.id])]),
      audit: admin ? await all(db, "SELECT action,target,actor_id,created_at FROM audit_events WHERE team_id=? ORDER BY created_at DESC LIMIT 50", teamId) : [],
    });
  }
  if (section === "members" && request.method === "POST") {
    if (!admin) fail("Administrator access required.", 403);
    const b = await body(request), target = text(b.accountId, "account"), role = b.role;
    if (!["admin","member","remove"].includes(role)) fail("Invalid role.");
    const statement = role === "remove"
      ? db.prepare("DELETE FROM team_members WHERE team_id=? AND account_id=? AND (role!='admin' OR (SELECT count(*) FROM team_members WHERE team_id=? AND role='admin')>1)").bind(teamId,target,teamId)
      : db.prepare("UPDATE team_members SET role=? WHERE team_id=? AND account_id=? AND (role!='admin' OR ?='admin' OR (SELECT count(*) FROM team_members WHERE team_id=? AND role='admin')>1)").bind(role,teamId,target,role,teamId);
    const r = role === "remove" ? (await db.batch([statement,db.prepare("UPDATE worker_keys SET revoked_at=? WHERE team_id=? AND account_id=? AND NOT EXISTS(SELECT 1 FROM team_members WHERE team_id=? AND account_id=?)").bind(now,teamId,target,teamId,target)]))[0] : await statement.run();
    if (!r.meta.changes) fail("Member not found, or this would remove the last administrator.", 409);
    await audit(db, teamId, user.id, `member.${role}`, target); return json({ok:true});
  }
  if (section === "invitations" && request.method === "POST") {
    if (!admin) fail("Administrator access required.", 403);
    await rate(db, "invites", teamId, 30, 3600);
    const invite = secret("dc_invite_"), expiresAt = now + 7 * 86400000;
    await db.prepare("INSERT INTO team_invitations(id,team_id,token_hash,created_by,expires_at) VALUES(?,?,?,?,?)").bind(crypto.randomUUID(),teamId,await digest(invite),user.id,expiresAt).run();
    await audit(db, teamId, user.id, "invite.created"); return json({invite,expiresAt}, 201);
  }
  if (section === "keys") {
    const b = await body(request);
    const target = b.accountId || user.id;
    if (!admin && target !== user.id) fail("You can manage only your own keys.", 403);
    await member(db, teamId, target);
    if (request.method === "POST" && !parts[3]) {
      const p = await project(db,teamId,b.projectId), token = secret("dc_worker_"), id = crypto.randomUUID();
      await rate(db,"issue-key",teamId,100,3600);
      const label = text(b.label || "My computer", "key label");
      await db.prepare("INSERT INTO worker_keys(id,team_id,project_id,account_id,token_hash,prefix,label,generation,created_at) VALUES(?,?,?,?,?,?,?,1,?)").bind(id,teamId,p.id,target,await digest(token),token.slice(0,18),label,now).run();
      await audit(db,teamId,user.id,"key.created",id); return json({id,key:token,generation:1},201);
    }
    if (request.method === "POST" && parts[3]) {
      const key = await db.prepare(`SELECT ${SAFE_KEY_FIELDS} FROM worker_keys WHERE id=? AND team_id=?`).bind(parts[3],teamId).first();
      if (!key) fail("Key not found.",404);
      if (!admin && key.account_id !== user.id) fail("You can manage only your own keys.",403);
      if (b.action === "revoke") {
        await db.prepare("UPDATE worker_keys SET revoked_at=? WHERE id=? AND team_id=?").bind(now,key.id,teamId).run();
        await audit(db,teamId,user.id,"key.revoked",key.id);return json({ok:true});
      }
      if (b.action !== "rotate" || !Number.isSafeInteger(b.generation)) fail("Specify the current key generation.");
      const token = secret("dc_worker_");
      const changed = await db.prepare("UPDATE worker_keys SET token_hash=?,prefix=?,generation=generation+1,created_at=? WHERE id=? AND team_id=? AND generation=? AND revoked_at IS NULL").bind(await digest(token),token.slice(0,18),now,key.id,teamId,b.generation).run();
      if (!changed.meta.changes) fail("Key changed or was revoked. Refresh before trying again.",409);
      await audit(db,teamId,user.id,"key.rotated",key.id);return json({id:key.id,key:token,generation:b.generation+1});
    }
    fail("Method not allowed.",405);
  }
  if (section === "projects") {
    if (!parts[3] && request.method === "POST") {
      if (!admin) fail("Administrator access required.",403);
      const spec = projectSpec(await body(request)), id = crypto.randomUUID();
      await db.prepare("INSERT INTO compute_projects(id,team_id,spec,revision,created_at) VALUES(?,?,?,1,?)").bind(id,teamId,JSON.stringify(spec),now).run();
      await audit(db,teamId,user.id,"project.created",id);return json({id,spec},201);
    }
    const p = await project(db,teamId,parts[3]);
    const kind = parts[4];
    if (["benchmark","rooms"].includes(kind)) {
      const actor = { team_id:teamId,project_id:p.id,account_id:user.id,userId:user.username,role:admin ? "coordinator" : "member" };
      return json(await coordinate(request, db, p, actor, kind));
    }
    if (request.method === "GET") return json({project:p, benchmark:await stateRead(db,p.id,"benchmark"),rooms:await stateRead(db,p.id,"rooms")});
    if (request.method === "POST" && !kind) {
      if (!admin) fail("Administrator access required.",403);
      const b = await body(request), spec = projectSpec(b);
      const bench = await stateRead(db,p.id,"benchmark"), rooms = await stateRead(db,p.id,"rooms");
      if (bench.session && !["finished","cancelled"].includes(bench.session.phase) || rooms.room && ["lobby","running"].includes(rooms.room.phase)) fail("Finish or cancel the current session before editing the model.",409);
      const r = await db.prepare("UPDATE compute_projects SET spec=?,revision=revision+1 WHERE id=? AND team_id=? AND revision=? AND NOT EXISTS(SELECT 1 FROM coordination_states WHERE project_id=?)").bind(JSON.stringify(spec),p.id,teamId,b.revision,p.id).run();
      if (!r.meta.changes) fail("Project changed. Refresh before saving.",409);
      await audit(db,teamId,user.id,"project.updated",p.id);return json({ok:true});
    }
  }
  fail("Endpoint not found.",404);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url), origin = request.headers.get("Origin");
    const allowed = !origin || origin === FRONTEND || origin === url.origin || /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(origin);
    let response;
    try {
      if (!allowed) fail("Origin not allowed.",403);
      if (request.method === "OPTIONS") response = new Response(null,{status:204});
      else if (url.pathname === "/api/health") response = json({service:"distributed-compute-platform",version:1,serverTime:Date.now()});
      else if (url.pathname === "/api/servers") response = json({servers:[]});
      else if (url.pathname.startsWith("/api/platform/")) response = await platform(request,env,url.pathname.slice(14).split("/").filter(Boolean));
      else if (["/api/benchmark","/api/rooms"].includes(url.pathname)) {
        const actor = await workerIdentity(request,env.DB), p = await project(env.DB,actor.team_id,actor.project_id);
        response = json(await coordinate(request,env.DB,p,actor,url.pathname.endsWith("rooms") ? "rooms" : "benchmark"));
      } else if (url.pathname === "/" && request.method === "GET") response = new Response('<!doctype html><html><head><title>Distributed Compute</title></head><body><h1>Distributed Compute coordination service</h1><p><a href="https://michaelkrymski.me/compute/">Open the platform</a></p></body></html>',{headers:{"content-type":"text/html;charset=utf-8"}});
      else fail("Endpoint not found.",404);
    } catch (error) {
      const status = Number.isInteger(error.status) ? error.status : 503;
      if (status === 503) console.error("Platform operation failed",url.pathname);
      response = json({error:status === 503 ? "Service unavailable. Please retry." : error.message},status);
    }
    const headers = new Headers(response.headers);
    if (origin && allowed) headers.set("Access-Control-Allow-Origin",origin);
    headers.set("Vary","Origin");headers.set("Access-Control-Allow-Headers","Authorization, Content-Type");headers.set("Access-Control-Allow-Methods","GET, POST, OPTIONS");headers.set("Cache-Control","no-store");headers.set("Referrer-Policy","no-referrer");headers.set("X-Content-Type-Options","nosniff");
    return new Response(response.body,{status:response.status,headers});
  },
};
