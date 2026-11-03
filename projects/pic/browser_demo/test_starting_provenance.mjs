import fs from 'node:fs';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('smoke_native_methods.mjs',import.meta.url),'utf8');
const start=source.indexOf('record.starting_suggestions?.some(');
assert.ok(start>=0);
let p=start+'record.starting_suggestions?.some('.length,depth=1,end=p;
for(;end<source.length;end++){if(source[end]==='(')depth++;if(source[end]===')'&&!--depth)break;}
const check=new Function('item','job','return ('+source.slice(p,end)+')(item)');
const job={method:'deepmod'};
const rule={method:'deepmod',version:'starting-settings-v1'};
const learned={method:'deepmod',version:'catboost-starting-settings-v1',learned:{engine:'CatBoost',applied:true,model_sha256:'a'.repeat(64)}};
assert.equal(check(rule,job),true);
assert.equal(check(learned,job),true);
assert.equal(check({...learned,learned:{...learned.learned,model_sha256:''}},job),false);
assert.equal(check({...learned,learned:{...learned.learned,applied:false}},job),false);
assert.equal(check({...learned,method:'epde'},job),false);
assert.equal(check({method:'deepmod',version:'unknown'},job),false);
console.log('RULE_AND_CATBOOST_PROVENANCE_PASS');
