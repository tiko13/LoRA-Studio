import test from 'node:test';
import assert from 'node:assert/strict';
import {connectionStatus} from '../web/connection-status.js';

const connection={backend:{available:true}};
const settings={integration:'promptstudio'};
const snapshot={source:'promptstudio',promptstudio:{enabled:true,reachable:true,configured:true},llm:{provider:'llamacpp',reachable:true,busy:false,vision:true,model:'synthetic'}};
test('connected Prompt Studio and working LLM have separate states',()=>{
  const view=connectionStatus({...snapshot,llm:{...snapshot.llm,busy:true,generation_phase:'thinking',generated_tokens:15}},connection,settings);
  assert.equal(view.rows.promptstudio.state,'idle');
  assert.equal(view.rows.llm.state,'busy');
  assert.equal(view.rows.llm.status,'Thinking · 15 tokens');
  assert.equal(view.tone,'busy');
});
test('LLM failure does not claim Prompt Studio disconnected',()=>{
  const view=connectionStatus({...snapshot,llm:{reachable:false}},connection,settings);
  assert.equal(view.rows.promptstudio.status,'Connected');
  assert.equal(view.rows.llm.state,'offline');
  assert.equal(view.tone,'offline');
});
test('standalone hides Prompt Studio and keeps unknown activity explicit',()=>{
  const view=connectionStatus({...snapshot,source:'standalone',llm:{...snapshot.llm,provider:'ollama',busy:null}},connection,{integration:'standalone'});
  assert.equal(view.rows.promptstudio.hidden,true);
  assert.equal(view.rows.llm.status,'Connected · activity unknown');
  assert.equal(view.rows.llm.state,'idle');
});
test('unknown vision is neutral, missing model and explicit no vision are red',()=>{
  assert.equal(connectionStatus({...snapshot,llm:{...snapshot.llm,vision:null}},connection,settings).rows.vision.state,'checking');
  assert.equal(connectionStatus({...snapshot,llm:{...snapshot.llm,vision:false}},connection,settings).rows.vision.state,'offline');
  assert.equal(connectionStatus({...snapshot,llm:{...snapshot.llm,model_installed:false}},connection,settings).rows.llm.state,'offline');
});
test('failed polling clears ready and working states',()=>{
  const view=connectionStatus(snapshot,connection,settings,'Disconnected');
  assert.equal(view.rows.llm.state,'offline');
  assert.equal(view.rows.promptstudio.state,'offline');
});
