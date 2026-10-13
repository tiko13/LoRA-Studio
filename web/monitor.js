import './image-preview.js';
import {escapeHTML as h} from './domain.js';

import {finite,ema,duration,number,bytes,chartSVG,metricsCSV,sampleIdentity} from './monitoring.js';

const $=s=>document.querySelector(s);
if(new URLSearchParams(location.search).get('embedded')==='1')document.body.classList.add('embedded-monitor');

let jobId=new URLSearchParams(location.search).get('job'), snapshot, pending=false, busy=false, gpuIndex=null;

let chartData=new Map(), samplesSignature='', renderedSamples='';

const fileURL=path=>`/api/jobs/${encodeURIComponent(jobId)}/file?path=${encodeURIComponent(path)}`;

async function api(path,body){const r=await fetch('/api'+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});const v=await r.json();if(!r.ok)throw Error(v.error||'Request failed');return v;}

function toast(message){$('#monitor-toast').textContent=message;$('#monitor-toast').hidden=false;setTimeout(()=>$('#monitor-toast').hidden=true,7000);}

function stat(label,value,note=''){return `<article class="stat"><span>${h(label)}</span><strong>${h(String(value))}</strong>${note?`<small>${h(note)}</small>`:''}</article>`;}

function setChart(id,points,key,options={}){const el=document.getElementById(id);if(!el)return;el.innerHTML=chartSVG(points,key,options);chartData.set(id,{points,key,options});}

function renderCharts(){if(!snapshot)return;const limit=Number($('#range').value),all=snapshot.points,points=limit?all.filter(p=>p.step>=snapshot.step-limit):all;

  setChart('loss-chart',points,'loss',{smooth:Number($('#smoothing').value),label:'Training loss'});setChart('lr-chart',points,'lr',{label:'Learning rate'});setChart('speed-chart',points,'seconds_per_step',{label:'Seconds per training step'});

  const hardware=snapshot.hardware_history.map(p=>({x:p.at,...p.gpus?.find(g=>g.index===gpuIndex)}));

  for(const [id,key,label] of [['memory','memory.used','VRAM · MiB'],['utilization','utilization.gpu','GPU utilization · %'],['temperature','temperature.gpu','Temperature · °C'],['power','power.draw','Power draw · W'],['bandwidth','utilization.memory','Memory utilization · %'],['clock','clocks.sm','SM clock · MHz']])setChart(id+'-chart',hardware,key,{time:true,label});

}

function renderSamples(){if(!snapshot)return;const files=snapshot.samples.filter(f=>f.path.toLowerCase().endsWith('.jpg')&&!f.path.toLowerCase().endsWith('.jpg.jpg')),chosen=files.length?files:snapshot.samples;

  const groups=new Map();for(const f of chosen){const identity=sampleIdentity(f.path);const key=identity.step===null?'Unlabelled':String(identity.step);if(!groups.has(key))groups.set(key,[]);groups.get(key).push({...f,...identity});}

  const keys=[...groups.keys()].sort((a,b)=>Number(a)-Number(b)),signature=keys.join(',');

  if(signature!==samplesSignature){for(const id of ['sample-before','sample-after']){const old=$('#'+id).value,follow=id==='sample-after'&&old===samplesSignature.split(',').at(-1);$('#'+id).innerHTML=keys.map(k=>`<option value="${h(k)}">${k==='0'?'Baseline':`Step ${h(k)}`}</option>`).join('');if(follow)$('#'+id).value=keys.at(-1);else if(keys.includes(old))$('#'+id).value=old;else $('#'+id).value=id==='sample-before'?keys[0]:keys.at(-1);}samplesSignature=signature;}

  const before=groups.get($('#sample-before').value)||[],after=groups.get($('#sample-after').value)||[];

  const renderKey=JSON.stringify([before,after]);if(renderKey===renderedSamples)return;renderedSamples=renderKey;

  $('#sample-grid').innerHTML=after.length?after.map((f,i)=>{const previous=before.find(b=>b.index===f.index)||before[i],cfg=snapshot.config.config?.process?.[0]?.sample||{},spec=cfg.samples?.[f.index]||{};return `<article class="card sample-comparison"><h3>Preview ${h(String(f.index))}</h3><div class="sample-pair">${[previous,f].map((img,k)=>img?`<figure><a href="${fileURL(img.path)}" data-image-preview><img loading="lazy" src="${fileURL(img.path)}" alt="Training preview ${h(String(img.index))}, step ${h(String(img.step))}"></a><figcaption>${k?'Current comparison':'Reference'} · ${img.step===0?'baseline':`step ${h(String(img.step??'?'))}`}</figcaption></figure>`:'').join('')}</div><details><summary>Prompt & generation settings</summary><p class="muted">${h(spec.prompt||'Prompt unavailable')}<br>${spec.width||cfg.width||'—'} × ${spec.height||cfg.height||'—'} · ${cfg.sample_steps||'—'} denoising steps · base seed ${cfg.seed??'—'}${cfg.walk_seed?' (walk seed)':''} · guidance ${cfg.guidance_scale??'—'}</p></details></article>`;}).join(''):'<div class="card muted">No previews saved yet. Baseline and checkpoint previews appear here automatically.</div>';

}

function renderLog(){if(!snapshot)return;const filter=$('#log-filter').value.toLowerCase();const el=$('#log'),scroll=el.scrollTop;el.textContent=filter?snapshot.log_tail.split('\n').filter(l=>l.toLowerCase().includes(filter)).join('\n'):snapshot.log_tail||'Waiting for trainer output…';el.scrollTop=$('#follow-log').checked?el.scrollHeight:scroll;}

function render(d){snapshot=d;const j=d.job,latest=d.latest||{},process=d.config.config?.process?.[0]||{};

  $('#run-title').textContent=j.snapshot?.name||'Training run';$('#run-subtitle').textContent=`${j.snapshot?.model||process.model?.arch||'Model'} · Started ${new Date(j.created*1000).toLocaleString()}${j.reconnected?' · Reconnected after Studio restart':''}`;

  $('#run-status').textContent=d.status;$('#run-status').className='badge '+(['running','complete'].includes(d.status)?'good':d.status==='failed'?'bad':'');

  $('#run-identity').innerHTML=`<small>Run ID</small><code>${h(j.id)}</code><small>Trainer PID · ${h(String(j.pid||'—'))}</small>`;

  $('#step').textContent=`${d.step.toLocaleString()} / ${d.total.toLocaleString()} steps · ${number(d.progress,1)}%`;

  $('#progress').value=d.progress;$('#phase').textContent=['running','pausing','queued'].includes(d.status)?d.phase:d.status;

  const controls=[['pause','Pause & save'],['resume','Resume checkpoint'],['save','Save checkpoint now'],['sample','Generate previews'],['cancel','Cancel run']];

  $('#controls').innerHTML=controls.map(([key,label])=>`<button data-control="${key}" ${!d.capabilities[key]||busy?'disabled':''} class="${key==='cancel'?'danger quiet':key==='resume'?'primary':''}">${label}</button>`).join('')+(d.status==='pausing'&&!d.control.stop_sent?'<button data-control="continue">Withdraw pause request</button>':'');

  $('#control-note').textContent=d.status==='pausing'?'Pause requested. Waiting for a complete checkpoint and optimizer save before releasing the GPU.':d.capabilities.native?'Pause requests a checkpoint and optimizer save, then releases the GPU. Cancel stops immediately; changes since the last checkpoint are lost.':'This existing run pauses at its next scheduled checkpoint. Cancel stops immediately; existing checkpoints are retained. Resume enables save-now controls.';
  $('#extension-form').hidden=d.status!=='complete';
  $('#extension-form button').disabled=busy||!d.capabilities.extend;
  renderExtension();

  $('#stats').innerHTML=stat('Training loss',number(latest.loss,5),'Latest reported step')+stat('Learning rate',finite(latest.lr)?latest.lr.toExponential(2):'—',process.train?.optimizer||'')+stat('Step duration',finite(latest.seconds_per_step)?number(latest.seconds_per_step,2)+' s':'—','Trainer rolling estimate')+stat('ETA',duration(d.eta_seconds),'Excludes future sampling / saving overhead')+stat('Wall time',duration(d.wall_seconds),'Includes startup and pauses')+stat('Last checkpoint',d.saved_step===null?'None':`Step ${d.saved_step}`,`${d.checkpoints.length} retained`)+stat('Checkpoint interval',`${process.save?.save_every??'—'} steps`,`Keep ${process.save?.max_step_saves_to_keep??'—'} checkpoints`)+stat('Preview interval',process.train?.disable_sampling?'Disabled':`${process.sample?.sample_every??'—'} steps`,`${process.sample?.sample_steps??'—'} denoising steps / image`)+stat('Next checkpoint',d.step<d.total?Math.min(d.total,(Math.floor(d.step/(process.save?.save_every||250))+1)*(process.save?.save_every||250)):'Finished')+stat('Next previews',process.train?.disable_sampling?'Disabled':d.step<d.total?Math.min(d.total,(Math.floor(d.step/(process.sample?.sample_every||250))+1)*(process.sample?.sample_every||250)):'Finished');

  $('#notices').innerHTML=(j.error?`<div class="note error">${h(j.error)}</div>`:'')+(d.invalid_values?`<div class="note warning">${d.invalid_values} non-finite metric reports detected. Inspect the log.</div>`:'')+(d.hardware.monitor_error?`<div class="note warning">Monitoring: ${h(d.hardware.monitor_error)}</div>`:'')+(d.log_updated&&Date.now()/1000-d.log_updated>120&&d.status==='running'?'<div class="note warning">No new log output for over two minutes. The trainer may be busy loading, saving, or stalled; this is not a failure diagnosis.</div>':'');

  const gpus=d.hardware.gpus||[];if(gpuIndex===null)gpuIndex=d.gpu_index??0;

  const optionSignature=gpus.map(g=>g.index+g.uuid).join();if($('#gpu-picker').dataset.signature!==optionSignature){$('#gpu-picker').innerHTML=gpus.map(g=>`<option value="${g.index}">GPU ${g.index} · ${h(g.name)}</option>`).join('');$('#gpu-picker').dataset.signature=optionSignature;$('#gpu-picker').value=String(gpuIndex);}

  const gpu=gpus.find(g=>g.index===gpuIndex)||{};

  $('#gpu-stats').innerHTML=stat('GPU load',finite(gpu['utilization.gpu'])?number(gpu['utilization.gpu'])+'%':'—')+stat('VRAM',finite(gpu['memory.used'])?`${number(gpu['memory.used']/1024,1)} / ${number(gpu['memory.total']/1024,1)} GiB`:'—')+stat('Temperature',finite(gpu['temperature.gpu'])?number(gpu['temperature.gpu'])+' °C':'—')+stat('Power',finite(gpu['power.draw'])?number(gpu['power.draw'])+' W':'—',finite(gpu['power.limit'])?'Limit '+number(gpu['power.limit'])+' W':'')+stat('Fan',finite(gpu['fan.speed'])?number(gpu['fan.speed'])+'%':'—')+stat('Telemetry samples',d.hardware_history.length.toLocaleString(),'Recorded every ~3 seconds');

  renderCharts();$('#metrics-table').innerHTML=d.points.slice(-300).reverse().map(p=>`<tr><td>${p.step}</td><td>${number(p.loss,6)}</td><td>${finite(p.lr)?p.lr.toExponential(3):'—'}</td><td>${number(p.seconds_per_step)}</td><td>${duration(p.elapsed)}</td></tr>`).join('');

  $('#diagnostics').innerHTML=`<p class="muted">${h(d.telemetry_note)} Table shows the latest 300 reports; CSV contains all available points.${d.history_truncated?' Log history exceeds the read limit; the oldest metrics may be absent.':''}</p>`;

  renderSamples();$('#saved-step').textContent=d.saved_step===null?'No checkpoint yet':`Last complete save · step ${d.saved_step}`;$('#resume-note').textContent=d.resume_note+(['paused','cancelled','failed','interrupted'].includes(d.status)&&d.resume_error?' '+d.resume_error:'');

  $('#checkpoint-table').innerHTML=d.checkpoints.slice().reverse().map(f=>`<tr><td>${h(f.path.split('/').at(-1))}</td><td>${bytes(f.size)}</td><td>${new Date(f.modified*1000).toLocaleString()}</td><td><a href="${fileURL(f.path)}">Download LoRA</a></td></tr>`).join('')||'<tr><td colspan="4">Checkpoints appear after the first save interval.</td></tr>';

  $('#file-list').innerHTML=d.files.map(f=>`<a href="${fileURL(f.path)}" target="_blank" rel="noopener">${h(f.path)} <small>${bytes(f.size)}</small></a>`).join('');

  $('#events').innerHTML=d.events.slice().reverse().map(e=>`<p><span class="badge">${h(e.kind)}</span> ${h(e.message)}</p>`).join('')+`<pre>${h(JSON.stringify(j.attempt_history||[],null,2))}</pre>`;

  renderLog();$('#download-log').href=fileURL('training.log');$('#download-config').href=fileURL('config.json');$('#config').textContent=JSON.stringify(d.config,null,2);

  const ds=d.dataset;$('#dataset-stats').innerHTML=stat('Frozen images',ds.count)+stat('Manual captions',ds.manual)+stat('Human reviewed',ds.reviewed)+stat('Caption warnings',ds.caption_warnings)+stat('Visual concerns',ds.visual_concerns)+stat('Batch / accumulation',`${process.train?.batch_size??'—'} / ${process.train?.gradient_accumulation??'—'}`);

  $('#backend-version').textContent='AI Toolkit revision: '+(d.backend_commit||'unknown')+' · Project edits do not alter these frozen inputs.';

}

function renderExtension(){
  if(!snapshot)return;
  const count=Number($('#additional-checkpoints').value),interval=snapshot.config.config?.process?.[0]?.save?.save_every;
  $('#extension-note').textContent=!snapshot.capabilities.extend?`Continuation unavailable. ${snapshot.resume_error||'A complete final checkpoint and optimizer save are required.'}`:
    Number.isInteger(count)&&count>=1&&count<=1000?`Add ${(count*interval).toLocaleString()} steps (${count} × ${interval.toLocaleString()}) to finish at step ${(snapshot.total+count*interval).toLocaleString()}. Uses this run’s frozen inputs and saved optimizer; previous outputs are retained.`:'Choose 1 to 1,000 additional checkpoint intervals.';
}
$('#additional-checkpoints').oninput=renderExtension;
$('#extension-form').onsubmit=async e=>{
  e.preventDefault();if(busy||!snapshot?.capabilities.extend)return;
  busy=true;$('#extension-form button').disabled=true;
  try{await api('/jobs/'+encodeURIComponent(jobId)+'/extend',{checkpoints:Number($('#additional-checkpoints').value)});toast('Continuing training from the final checkpoint.');}
  catch(error){toast(error.message);}finally{busy=false;await refresh();}
};

async function refresh(){if(pending||!jobId)return;pending=true;try{const d=await api('/jobs/'+encodeURIComponent(jobId)+'/monitor');render(d);$('#error').hidden=true;$('#connection').textContent='Updated '+new Date().toLocaleTimeString();}catch(e){$('#error').textContent=e.message;$('#error').hidden=false;$('#connection').textContent='Disconnected · retrying';}finally{pending=false;}}

function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}

$('#refresh').onclick=refresh;$('#range').onchange=$('#smoothing').onchange=renderCharts;

$('#gpu-picker').onchange=()=>{gpuIndex=Number($('#gpu-picker').value);render(snapshot);};$('#sample-before').onchange=$('#sample-after').onchange=renderSamples;$('#log-filter').oninput=renderLog;

$('#export-csv').onclick=()=>snapshot&&download('training-metrics.csv',metricsCSV(snapshot.points),'text/csv');$('#export-json').onclick=()=>snapshot&&download('training-telemetry.json',JSON.stringify(snapshot,null,2),'application/json');

document.addEventListener('click',async e=>{const b=e.target.closest('[data-control]');if(!b||busy)return;busy=true;b.disabled=true;try{await api('/jobs/'+encodeURIComponent(jobId)+'/'+b.dataset.control,{});toast('Training control request accepted.');await refresh();}catch(err){toast(err.message);}finally{busy=false;await refresh();}});

document.addEventListener('pointermove',e=>{const svg=e.target.closest('.metric-chart');const tip=$('#chart-tooltip');if(!svg){tip.hidden=true;return;}const source=chartData.get(svg.parentElement.id);if(!source)return;const rect=svg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((e.clientX-rect.left)/rect.width*800-65)/715));const x=Number(svg.dataset.minX)+fraction*(Number(svg.dataset.maxX)-Number(svg.dataset.minX));const rows=source.points.filter(p=>finite(p[source.key]));if(!rows.length)return;const p=rows.reduce((a,b)=>Math.abs((a.x??a.step)-x)<Math.abs((b.x??b.step)-x)?a:b);tip.textContent=`${p.x?new Date(p.x*1000).toLocaleTimeString():'Step '+p.step} · ${number(p[source.key],6)}`;tip.style.left=Math.min(innerWidth-230,e.clientX+12)+'px';tip.style.top=Math.max(8,e.clientY-38)+'px';tip.hidden=false;});

$('#run-picker').onchange=()=>{location.search='?job='+encodeURIComponent($('#run-picker').value)+(document.body.classList.contains('embedded-monitor')?'&embedded=1':'');};

async function boot(){const b=await api('/bootstrap'),runs=[];for(const p of b.projects){const d=await api('/projects/'+p.id);runs.push(...d.jobs.filter(j=>j.kind==='train').map(j=>({...j,projectName:p.name})));}runs.sort((a,b)=>b.created-a.created);$('#run-picker').innerHTML=runs.map(j=>`<option value="${h(j.id)}">${h(j.projectName)} · ${h(j.status)} · ${new Date(j.created*1000).toLocaleTimeString()}</option>`).join('');jobId=jobId||runs[0]?.id;if(!jobId){$('#run-title').textContent='No training runs yet';$('#connection').textContent='Connected';return;}$('#run-picker').value=jobId;history.replaceState(null,'','?job='+encodeURIComponent(jobId)+(document.body.classList.contains('embedded-monitor')?'&embedded=1':''));$('#hardware-charts').innerHTML=[['memory','VRAM · MiB'],['utilization','GPU utilization · %'],['temperature','Temperature · °C'],['power','Power · W'],['bandwidth','Memory utilization · %'],['clock','SM clock · MHz']].map(([id,label])=>`<article class="card chart-card"><h3>${label}</h3><div id="${id}-chart"></div></article>`).join('');await refresh();}

boot().catch(e=>{$('#error').textContent=e.message;$('#error').hidden=false;});setInterval(()=>{if(!document.hidden)refresh();},3000);


$('#preset-form').onsubmit=async e=>{e.preventDefault();const form=e.currentTarget,button=form.querySelector('button');button.disabled=true;try{const preset=await api('/jobs/'+encodeURIComponent(jobId)+'/preset',{name:form.elements.name.value,default:form.elements.default.checked});$('#preset-result').textContent='Saved '+preset.name+(preset.default?' as the default for new '+preset.model+' projects.':'.');}catch(error){$('#preset-result').textContent=error.message;}finally{button.disabled=false;}};
