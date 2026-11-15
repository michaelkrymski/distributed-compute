// Run locally once. Never commit the generated plaintext key file or .dev.vars.
import {randomBytes,createHash} from 'node:crypto';
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
const path='work/team-access-keys.json';
if(existsSync(path))throw new Error('Keys already exist; preserve them or explicitly rotate with the hosting owner.');
const roster=['onlyspxctre','michaelkrymski','jsaluja09','jacklowry08','carlimarquilage-ctrl'];
const plaintext={},hashed={};
for(const id of roster){const key='glm_'+randomBytes(32).toString('base64url');plaintext[id]=key;hashed[id]={sha256:createHash('sha256').update(key).digest('hex'),role:id==='michaelkrymski'?'coordinator':'member'};}
mkdirSync('work',{recursive:true});
writeFileSync(path,JSON.stringify(plaintext,null,2)+'\n',{mode:0o600});
writeFileSync('work/team-key-hashes.json',JSON.stringify(hashed),{mode:0o600});
writeFileSync('.dev.vars',"TEAM_KEYS_JSON='"+JSON.stringify(hashed)+"'\n",{mode:0o600});
console.log('Created private team keys under work/ and local .dev.vars. No keys were printed.');
