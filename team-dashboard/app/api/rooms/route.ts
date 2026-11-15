import { env } from "cloudflare:workers";
import { getTeamUser, apiJson, apiOptions } from "../../../lib/access";
export const dynamic = "force-dynamic";
export const OPTIONS = apiOptions;
export async function GET(request: Request) {
 const user = await getTeamUser(request);
 if (!user) return apiJson({error:"Authentication required"},401);
 if (!env.DB) return apiJson({error:"Database unavailable"},503);
 const room = await env.DB.prepare("SELECT * FROM training_rooms ORDER BY created_at DESC LIMIT 1").first();
 const members = room ? (await env.DB.prepare("SELECT * FROM training_members WHERE room_id=? ORDER BY instance_id").bind(room.id).all()).results : [];
 return apiJson({serverTime:Date.now(),user:{id:user.userId,role:user.role},room,members});
}
export async function POST(request: Request) {
 const user = await getTeamUser(request);
 if (!user) return apiJson({error:"Authentication required"},401);
 const raw = await request.text();
 if(raw.length>16000) return apiJson({error:"Request too large"},413);
 let b; try { b=JSON.parse(raw); } catch { return apiJson({error:"Invalid JSON"},400); }
 if(!b || typeof b!=="object") return apiJson({error:"Invalid request"},400);
 if (!env.DB) return apiJson({error:"Database unavailable"},503);
 const now=Date.now(), db=env.DB;
 const admin=user.role==="coordinator";
 if(b.action==="create") {
  if(!admin) return apiJson({error:"Organizer required"},403);
  if(typeof b.name!=="string" || b.name.length<1 || b.name.length>100 || !/^[a-f0-9]{64}$/.test(b.fingerprint) || typeof b.peer!=="string" || !/^\/ip[46]\/[^\s]+\/tcp\/\d+\/p2p\/[A-Za-z0-9]+$/.test(b.peer) || !Number.isSafeInteger(b.targetTokens) || b.targetTokens<1 || !Number.isInteger(b.shards) || b.shards<1 || b.shards>1000) return apiJson({error:"Verify bundle, target, shards and peer address"},400);
  try { await db.prepare("INSERT INTO training_rooms(id,active_slot,name,fingerprint,peer,target_tokens,shards,phase,created_by,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)").bind(crypto.randomUUID(),"active",b.name,b.fingerprint,b.peer,b.targetTokens,b.shards,"lobby",user.userId,now).run(); }
  catch { return apiJson({error:"A training room already exists"},409); }
  return apiJson({ok:true});
 }
 const room=await db.prepare("SELECT * FROM training_rooms WHERE id=?").bind(b.roomId??"").first();
 if(!room) return apiJson({error:"Room not found"},404);
 if(b.action==="cancel") {
  if(!admin) return apiJson({error:"Organizer required"},403);
  await db.prepare("UPDATE training_rooms SET phase='cancelled',active_slot=NULL WHERE id=?").bind(room.id).run();return apiJson({ok:true});
 }
 if(b.action==="start") {
  if(!admin) return apiJson({error:"Organizer required"},403);
  const results=await db.batch([db.prepare("UPDATE training_rooms SET phase='running', expected=(SELECT json_group_array(instance_id) FROM training_members WHERE room_id=? AND last_seen>?) WHERE id=? AND phase='lobby' AND EXISTS(SELECT 1 FROM training_members WHERE room_id=? AND last_seen>?) AND NOT EXISTS(SELECT 1 FROM training_members WHERE room_id=? AND last_seen>? AND ready=0)").bind(room.id,now-20000,room.id,room.id,now-20000,room.id,now-20000), db.prepare("UPDATE training_members SET shard=(SELECT count(*) FROM json_each((SELECT expected FROM training_rooms WHERE id=?)) WHERE value<training_members.instance_id), leader=CASE WHEN instance_id=(SELECT min(value) FROM json_each((SELECT expected FROM training_rooms WHERE id=?))) THEN 1 ELSE 0 END WHERE room_id=? AND instance_id IN (SELECT value FROM json_each((SELECT expected FROM training_rooms WHERE id=?)))").bind(room.id,room.id,room.id,room.id)]);
  if(!results[0].meta.changes) return apiJson({error:"Every online computer must be ready"},409);
  return apiJson({ok:true});
 }
 if(typeof b.instanceId!=="string" || !/^[a-f0-9]{32}$/.test(b.instanceId)) return apiJson({error:"Invalid computer ID"},400);
 if(b.action==="join") {
  if(b.fingerprint!==room.fingerprint || typeof b.name!=="string" || b.name.length>100 || typeof b.device!=="string" || b.device.length>200) return apiJson({error:"Run bundle mismatch or invalid computer"},400);
  const r=await db.prepare("INSERT INTO training_members(room_id,instance_id,user_id,name,device,fingerprint,last_seen) SELECT ?,?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM training_rooms WHERE id=? AND phase='lobby') AND (SELECT count(*) FROM training_members WHERE room_id=?) < ? ON CONFLICT DO NOTHING").bind(room.id,b.instanceId,user.userId,b.name,b.device,b.fingerprint,now,room.id,room.id,room.shards).run();
  return r.meta.changes ? apiJson({ok:true}) : apiJson({error:"Room started, full, or computer already joined"},409);
 }
 if(b.action==="leave") {
  await db.prepare("DELETE FROM training_members WHERE room_id=? AND instance_id=? AND user_id=? AND EXISTS(SELECT 1 FROM training_rooms WHERE id=? AND phase='lobby')").bind(room.id,b.instanceId,user.userId,room.id).run();
  return apiJson({ok:true});
 }
 if(b.action==="ready") {
  if(typeof b.ready!=="boolean") return apiJson({error:"Invalid ready value"},400);
  const r=await db.prepare("UPDATE training_members SET ready=? WHERE room_id=? AND instance_id=? AND user_id=? AND EXISTS(SELECT 1 FROM training_rooms WHERE id=? AND phase='lobby')").bind(b.ready?1:0,room.id,b.instanceId,user.userId,room.id).run();
  return r.meta.changes ? apiJson({ok:true}) : apiJson({error:"Computer not owned or room started"},409);
 }
 if(b.action==="heartbeat") {
  const progress= b.progress && typeof b.progress==="object" ? JSON.stringify(b.progress).slice(0,4000) : null;
  const r=await db.prepare("UPDATE training_members SET last_seen=?,progress=coalesce(?,progress) WHERE room_id=? AND instance_id=? AND user_id=?").bind(now,progress,room.id,b.instanceId,user.userId).run();
  if(b.progress?.event==="worker_finished" && r.meta.changes) {
   await db.prepare("UPDATE training_rooms SET phase='finished',active_slot=NULL WHERE id=? AND phase='running' AND NOT EXISTS(SELECT 1 FROM training_members WHERE room_id=? AND instance_id IN (SELECT value FROM json_each(training_rooms.expected)) AND coalesce(json_extract(progress,'$.event'),'')!='worker_finished')").bind(room.id,room.id).run();
  }
  return r.meta.changes ? apiJson({ok:true}) : apiJson({error:"Computer not owned"},403);
 }
 return apiJson({error:"Unknown action"},400);
}
