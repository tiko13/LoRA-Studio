import test from 'node:test';
import assert from 'node:assert/strict';
import {escapeHTML,imageState,resolvedAttributes} from '../web/domain.js';
test('user supplied text cannot become markup',()=>{
  assert.equal(escapeHTML('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
});
test('stale captions cannot appear ready even after previous review',()=>{
  assert.equal(imageState({included:true,caption:'target',reviewed:true,intent_digest:'old'},'new').label,'Rules changed');
});
test('excluded and warning captions have distinct states',()=>{
  assert.equal(imageState({included:false},'x').label,'Excluded');
  assert.equal(imageState({included:true,caption:'x',intent_digest:'x',warnings:['unclear'],reviewed:false},'x').label,'Review needed');
});
test('Auto follows current type without removing hidden choices',()=>{
  const p={kind:'style',choices:{hair:'keep',medium:'auto'}};
  const catalog={types:[{id:'style',attributes:[{id:'medium',default:'keep'}]}]};
  assert.equal(resolvedAttributes(p,catalog)[0].resolved,'keep');
  assert.equal(p.choices.hair,'keep');
});
