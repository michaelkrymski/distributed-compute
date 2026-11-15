// Local-only real API fixture. Credentials are saved privately, never printed.
import {randomBytes} from "node:crypto";
import {mkdir,writeFile} from "node:fs/promises";
const base=process.env.COMPUTE_TEST_API || "http://127.0.0.1:5195";
if(!["127.0.0.1","localhost"].includes(new URL(base).hostname))throw Error("Local only");
async function post(path,b,token){const r=await fetch(base+path,{method:"POST",headers:{"Content-Type":"application/json",...(token?{Authorization:"Bearer "+token}:{})},body:JSON.stringify(b)});const d=await r.json();if(!r.ok)throw Error(d.error);return d;}
const model={vocab_size:257,width:32,layers:1,heads:4,expansion:2,context:64,tied:true};
const account=await post("/api/platform/setup",{username:"cpu_"+randomBytes(5).toString("hex"),password:randomBytes(24).toString("hex"),teamName:"CPU acceptance fixture",project:{name:"Tiny decoder",model,targetTokens:8192}});
const key=await post(`/api/platform/teams/${account.teamId}/keys`,{projectId:account.projectId,label:"CPU test"},account.token);
await mkdir("work",{recursive:true});
await writeFile("work/cpu-worker-keys.json",JSON.stringify({michaelkrymski:key.key}),{mode:0o600});
console.log("Created local CPU-worker fixture; no keys printed.");
