// Actual HTTP and local D1. Registration fixtures do not represent training.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {randomBytes} from 'node:crypto';
const base=process.env.GLM_API_BASE??'http://127.0.0.1:5181';
if(!['127.0.0.1','localhost'].includes(new URL(base).hostname)) throw Error('Local only');
const keys=JSON.parse(fs.readFileSync('work/team-access-keys.json','utf8'));
const owner='michaelkrymski', member=Object.keys(keys).find(id=>id!==owner);
async function call(user,body,status=200) {
 const r=await fetch(base+'/api/rooms',{method:body?'POST':'GET',headers:{Authorization:'Bearer '+(keys[user]??''),'content-type':'application/json'},body:body?JSON.stringify(body):undefined});
 const data=await r.json();assert.equal(r.status,status,JSON.stringify(data));return data;
}
await call('none',null,401);
const config={action:'create',name:'Local fixture',fingerprint:'a'.repeat(64),targetTokens:10000,shards:2,peer:'/ip4/127.0.0.1/tcp/40101/p2p/QmTest'};
await call(member,config,403);
const previous=(await call(owner)).room;
if(previous && previous.phase!=='cancelled')await call(owner,{action:'cancel',roomId:previous.id});
await call(owner,config);
const roomId=(await call(owner)).room.id;
await call(owner,{action:'start',roomId},409);
const instances=[randomBytes(16).toString('hex'),randomBytes(16).toString('hex')];
for(const instanceId of instances)await call(member,{action:'join',roomId,instanceId,name:'Fixture',device:'Not a training worker',fingerprint:'a'.repeat(64)});
await call(owner,{action:'ready',roomId,instanceId:instances[0],ready:true},409);
await call(member,{action:'ready',roomId,instanceId:instances[0],ready:true});
await call(owner,{action:'start',roomId},409);
await call(member,{action:'ready',roomId,instanceId:instances[1],ready:true});
await call(member,{action:'start',roomId},403);
await call(owner,{action:'start',roomId});
const state=await call(owner);
assert.equal(state.room.phase,'running');
assert.equal(state.members.filter(m=>m.leader).length,1);
assert.deepEqual(state.members.map(m=>m.shard).sort(),[0,1]);
await call(member,{action:'join',roomId,instanceId:randomBytes(16).toString('hex'),name:'Late',device:'CPU',fingerprint:'a'.repeat(64)},409);
await call(owner,{action:'cancel',roomId});
console.log('Room HTTP/D1 acceptance passed');

const passwordFile='work/native-test-password.json';
const password=fs.existsSync(passwordFile)?JSON.parse(fs.readFileSync(passwordFile,'utf8')).password:randomBytes(24).toString('base64url');
fs.writeFileSync(passwordFile,JSON.stringify({password}),{mode:0o600});
const setup=await fetch(base+'/api/auth',{method:'POST',headers:{Authorization:'Bearer '+keys[owner],Origin:base,'content-type':'application/json'},body:JSON.stringify({action:'setup',password})});
assert.ok([200,409].includes(setup.status),'Local account activation failed');
const login=await fetch(base+'/api/desktop-auth',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({password})});
assert.equal(login.status,200);const auth=await login.json();
const native=await fetch(base+'/api/rooms',{headers:{Authorization:'Bearer '+auth.token}});
assert.equal(native.status,200);assert.equal((await native.json()).user.role,'coordinator');
console.log('Native admin password exchange and device-token authorization passed');
