// Re-runnable local owner wizard. Never prints credentials or overwrites other runtime settings.
import {createHash,randomBytes} from "node:crypto";
import {existsSync,readFileSync,writeFileSync,mkdirSync} from "node:fs";
import {createInterface} from "node:readline/promises";
const input=createInterface({input:process.stdin,output:process.stdout});
const read=(p)=>existsSync(p)?JSON.parse(readFileSync(p,"utf8")):{};
const hashes=read("work/team-key-hashes.json"), keys=read("work/team-access-keys.json");
try {
 console.log("GLM team setup. Keep work/ private. Existing members remain unchanged unless selected.");
 const action=(await input.question("Action [add / rotate / revoke / role]: ")).trim();
 const id=(await input.question("Member account ID: ")).trim();
 if(!/^[a-zA-Z0-9_-]{1,80}$/.test(id)) throw Error("Use a simple account ID");
 if(action==="revoke") {delete hashes[id];delete keys[id];}
 else if(action==="role") {
  if(!hashes[id])throw Error("Unknown member");
  const role=(await input.question("Role [member / coordinator]: ")).trim();
  if(!["member","coordinator"].includes(role))throw Error("Invalid role");hashes[id].role=role;
 } else if(action==="add" || action==="rotate") {
  if(action==="add" && hashes[id])throw Error("Member exists. Choose rotate explicitly");
  const role=(await input.question("Role [member / coordinator]: ")).trim();
  if(!["member","coordinator"].includes(role))throw Error("Invalid role");
  const key="glm_"+randomBytes(32).toString("base64url");keys[id]=key;hashes[id]={sha256:createHash("sha256").update(key).digest("hex"),role};
 } else throw Error("Unknown action");
 mkdirSync("work",{recursive:true});
 writeFileSync("work/team-access-keys.json",JSON.stringify(keys,null,2),{mode:0o600});
 writeFileSync("work/team-key-hashes.json",JSON.stringify(hashes),{mode:0o600});
 const vars=existsSync(".dev.vars")?readFileSync(".dev.vars","utf8"):"";
 const line="TEAM_KEYS_JSON='"+JSON.stringify(hashes)+"'";
 writeFileSync(".dev.vars",vars.match(/^TEAM_KEYS_JSON=.*$/m)?vars.replace(/^TEAM_KEYS_JSON=.*$/m,line):vars+"\n"+line+"\n",{mode:0o600});
 console.log("Saved private local keys and hash configuration. Update the backend TEAM_KEYS_JSON secret using work/team-key-hashes.json; local edits alone do not change production. Share only each member's own key privately.");
} finally {input.close();}
