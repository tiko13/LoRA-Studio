const providers={llamacpp:'Llama.cpp',ollama:'Ollama',koboldcpp:'KoboldCpp',openai:'OpenAI-compatible'};
const row=(label,state,status,detail='')=>({label,state,status,detail});

export function connectionStatus(snapshot,connection,settings={},error='') {
  const llm=snapshot?.llm, ps=snapshot?.promptstudio;
  const dedicated=settings.caption_source&&settings.caption_source!=='connected';
  const reachable=llm?.reachable===true;
  const missing=llm?.model_installed===false;
  const failed=Boolean(error||llm?.handoff_error||llm&&(!reachable||missing));
  const busy=reachable&&llm.busy===true;
  const llmState=failed?'offline':busy?'busy':reachable?'idle':'checking';
  const phase={thinking:'Thinking',generating:'Generating',thinking_or_generating:'Thinking / generating'}[llm?.generation_phase]||'Processing';
  const count=Number.isFinite(llm?.generated_tokens)?` · ${llm.generated_tokens.toLocaleString()} tokens`:'';
  const llmLabel=failed?(missing?'Model unavailable':'Unavailable'):busy?(llm.model_state==='loading'?'Loading model':phase)+count:reachable?(llm.busy===false?'Ready':'Connected · activity unknown'):'Checking…';
  const source=snapshot?.source==='promptstudio'?'Via Prompt Studio':'Standalone';
  const detail=error||llm?.handoff_error|| (missing?'Select an available model in Backend settings.':llm?.message)||'';
  const rows={
    promptstudio:row('Prompt Studio',error?'offline':!ps?'checking':ps.reachable&&ps.configured?'idle':'offline',
      error?'Status unavailable':!ps?'Checking…':ps.reachable?(ps.configured?'Connected':'Needs setup'):'Disconnected',ps?.message||'Uses the saved connection in the active chat.'),
    llm:row(providers[llm?.provider]||'Backend LLM',llmState,llmLabel,
      [source,llm?.model,detail].filter(Boolean).join(' · ')),
  };
  rows.promptstudio.hidden=settings.integration==='standalone';
  const vision=llm?.vision;
  rows.vision=dedicated
    ? row('Captioning',connection?.captioner?.configured?'idle':connection?'offline':'checking',connection?.captioner?.configured?'Configured':connection?'Needs setup':'Checking…',connection?.captioner?.reason||'Dedicated captioning backend; runtime readiness is checked when used.')
    : row('Vision support',failed?'offline':vision===true?'idle':vision===false?'offline':'checking',failed?'Unavailable':vision===true?'Supported':vision===false?'Not supported':'Unknown',
      failed?'Restore the LLM connection to check vision.':vision===true?'Image support reported by the selected backend.':vision===false?'Select a vision-capable model for captioning.':'Image support has not been confirmed.');
  rows.toolkit=row('AI Toolkit',connection?.backend?.available?'idle':connection?'offline':'checking',connection?.backend?.available?'Available':connection?'Needs setup':'Checking…',connection?.backend?.available?'Installation available. Model support is checked before training.':connection?.backend?.reason||'');
  const visible=Object.values(rows).filter(r=>!r.hidden);
  const tone=visible.some(r=>r.state==='offline')?'offline':visible.some(r=>r.state==='busy')?'busy':visible.some(r=>r.state==='checking')?'checking':'idle';
  return {rows,tone,summary:{offline:'Connections need attention',busy:'LLM is working',checking:'Some status is still unknown',idle:'Connections ready'}[tone]};
}

export function startConnectionMonitor({api,getSettings,onChange}) {
  let pending=null,last=0,snapshot=null,error='';
  async function refresh() {
    if(pending)return pending;
    if(!getSettings())return;
    const revision=getSettings().settings_revision;
    pending=(async()=>{
      try {
        const response=await api('/connection/status');
        if(getSettings().settings_revision!==revision||response.settings_revision!==revision)return;
        snapshot=response;error='';last=Date.now();
      } catch(e) {if(getSettings().settings_revision===revision){snapshot=null;error=e.message;last=Date.now();}}
      finally {pending=null;onChange();}
    })();
    return pending;
  }
  setInterval(()=>{
    if(!document.hidden&&Date.now()-last>(document.querySelector('#system-status').open?4000:14000))refresh();
  },1000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
  return {refresh,get snapshot(){return snapshot?.settings_revision===getSettings()?.settings_revision?snapshot:null;},get error(){return error;},get checked(){return last;}};
}
