import { authenticate } from "../auth/route";
export async function POST(request: Request) {
 const raw=await request.text();
 if(raw.length>4096) return Response.json({error:"Request too large"},{status:413});
 let b;try{b=JSON.parse(raw);}catch{return Response.json({error:"Invalid JSON"},{status:400});}
 if(!b || typeof b!=="object")return Response.json({error:"Invalid request"},{status:400});
 return authenticate(new Request(request.url,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({action:"login",password:b.password})}),true);
}
