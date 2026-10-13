import test from 'node:test';
import assert from 'node:assert/strict';
import {moveCrop,resizeCrop,fitCropRatio} from '../web/crop-geometry.js';

test('moving crops clamps to the image without changing their size',()=>{
  const result=moveCrop([.2,.3,.7,.9],2,-2);
  assert.deepEqual(result,[.5,0,1,.6000000000000001]);
});
test('each free edge expands independently and cannot cross its opposite',()=>{
  assert.deepEqual(resizeCrop([.2,.2,.8,.8],'w',-.1,0),[.1,.2,.8,.8]);
  assert.deepEqual(resizeCrop([.2,.2,.8,.8],'s',0,1),[.2,.2,.8,1]);
  const r=resizeCrop([.2,.2,.8,.8],'nw',1,1,0,.01,.01);
  assert(r[0]<r[2]&&r[1]<r[3]);
});
test('locked ratio stays exact and in bounds for all eight handles',()=>{
  for(const ratio of [.25,1,4])for(const handle of ['n','ne','e','se','s','sw','w','nw']){
    const rect=fitCropRatio([.1,.1,.9,.9],ratio);
    for(const dx of [-2,-.1,.1,2])for(const dy of [-2,-.1,.1,2]){
      const r=resizeCrop(rect,handle,dx,dy,ratio);
      assert(r.every(v=>v>=-1e-10&&v<=1+1e-10));
      assert(Math.abs((r[2]-r[0])/(r[3]-r[1])-ratio)<1e-8);
    }
  }
});
