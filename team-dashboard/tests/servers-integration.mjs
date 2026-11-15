// Actual local HTTP/D1 acceptance. Never run against production.
import assert from 'node:assert/strict';
import fs from 'node:fs';
const base=process.env.GLM_API_BASE??'http://127.0.0.1:5181';
if(!['127.0.0.1','localhost'].includes(new URL(base).hostname))throw Error('Local only');
const keys=JSON.parse(fs.readFileSync('work/team-access-keys.json','utf8'));
const owner='michaelkrymski',member=Object.keys(keys).find(id=>id!==owner);
async function directory(){const r=await fetch(base+'/api/servers');assert.equal(r.status,200);return (await r.json()).servers;}
async function save(id,body,status){
 const r=await fetch(base+'/api/servers',{method:'POST',headers:{Authorization:'Bearer '+(keys[id]??''),'Content-Type':'application/json'},body:JSON.stringify(body)});
 assert.equal(r.status,status,await r.text());
}
const fixture={name:'Local directory fixture',description:'Metadata only',origin:'https://directory-fixture.invalid',isPublic:false};
await save('none',fixture,403);
await save(member,fixture,403);
await save(owner,fixture,200);
assert.ok(!(await directory()).some(row=>row.origin===fixture.origin));
await save(owner,{...fixture,isPublic:true},200);
assert.ok((await directory()).some(row=>row.origin===fixture.origin));
for(const row of await directory())assert.deepEqual(Object.keys(row).sort(),['description','id','name','origin']);
await save(owner,{...fixture,origin:'https://user:secret@example.com'},400);
await save(owner,{...fixture,origin:'http://example.com'},400);
await save(owner,{...fixture,origin:'https://example.com/path'},400);
await save(owner,{...fixture,isPublic:false},200);
assert.ok(!(await directory()).some(row=>row.origin===fixture.origin));
console.log('Real directory HTTP/D1 checks passed: public metadata, curator-only writes, private exclusion, URL validation and unlisting');
