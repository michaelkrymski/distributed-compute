import { env } from "cloudflare:workers";
import { getTeamUser, apiJson, apiOptions } from "../../../lib/access";
export const dynamic = "force-dynamic";
export const OPTIONS = apiOptions;
// Public metadata only. Never query team/session/device tables here.
export async function GET() {
 if(!env.DB) return apiJson({error:"Directory unavailable"},503);
 const result=await env.DB.prepare("SELECT id,name,description,origin FROM server_directory WHERE is_public=1 ORDER BY name LIMIT 50").all();
 return apiJson({servers:result.results});
}
export async function POST(request: Request) {
 const user=await getTeamUser(request);
 if(!user || user.role!=="coordinator") return apiJson({error:"Directory curator required"},403);
 if(!env.DB) return apiJson({error:"Directory unavailable"},503);
 const raw=await request.text();if(raw.length>4096)return apiJson({error:"Request too large"},413);
 let b;try{b=JSON.parse(raw);}catch{return apiJson({error:"Invalid JSON"},400);}
 if(!b || typeof b!=="object" || typeof b.name!=="string" || b.name.trim().length<1 || b.name.length>80 || typeof b.description!=="string" || b.description.length>240 || typeof b.origin!=="string" || b.origin.length>250 || typeof b.isPublic!=="boolean")return apiJson({error:"Invalid listing"},400);
 let origin;
 try {
  const u=new URL(b.origin);
  if(u.protocol!=="https:" || u.username || u.password || u.search || u.hash || (u.pathname!=="/" && u.pathname!==""))throw Error();
  origin=u.origin;
 } catch {return apiJson({error:"Use an HTTPS server origin without a path or credentials"},400);}
 await env.DB.prepare("INSERT INTO server_directory(id,name,description,origin,is_public,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(origin) DO UPDATE SET name=excluded.name,description=excluded.description,is_public=excluded.is_public,updated_at=excluded.updated_at").bind(crypto.randomUUID(),b.name.trim(),b.description,origin,b.isPublic?1:0,Date.now()).run();
 return apiJson({ok:true});
}
