import {escapeHTML as h, imageState, resolvedAttributes} from './domain.js';

const $ = s => document.querySelector(s);
const state = {projects:[], catalog:null, settings:null, current:null, tab:'setup', dirty:false, saving:null, timer:null, connection:null, recovered:false};
const icons = {character:'◎',object:'◇',style:'✧',concept:'↔'};
const draftKey = id => `lora-studio.draft.${id}`;
let toastTimer;
function toast(message, error=false) { const el=$('#toast'); el.textContent=message; el.className=error?'error':''; el.hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true,error?12000:4500); }
async function api(path, method='GET', value) { const response=await fetch('/api'+path,{method,headers:value?{'Content-Type':'application/json'}:{},body:value?JSON.stringify(value):undefined}); const data=await response.json(); if(!response.ok) throw new Error(data.error||'Request failed'); return data; }
function saveLabel(text) { $('#save-state').textContent=text; }
function localDraft() { try { localStorage.setItem(draftKey(state.current.project.id),JSON.stringify(state.current.project)); } catch { toast('Browser draft storage is unavailable. Keep this tab open until saved.',true); } }
function changed() { state.dirty=true; localDraft(); saveLabel('Saving…'); clearTimeout(state.timer); state.timer=setTimeout(()=>save().catch(e=>toast(e.message,true)),550); renderInspector(); }
async function save() {
  clearTimeout(state.timer);
  if(state.saving) { await state.saving; if(state.dirty) return save(); return; }
  if(!state.dirty || !state.current) return;
  const snapshot=structuredClone(state.current.project);
  state.dirty=false;
  state.saving=(async()=>{
    try {
      const result=await api('/projects/'+snapshot.id,'PUT',snapshot);
      if(state.current.project.id!==result.id) return;
      state.current.project.revision=result.revision;
      if(state.dirty) localDraft(); else { state.current.project=result; localStorage.removeItem(draftKey(result.id)); }
      state.projects=state.projects.map(p=>p.id===result.id?result:p);
      $('#project-title').textContent=state.current.project.name||'Untitled LoRA';
      renderProjects(); saveLabel(state.dirty?'Saving…':'All changes saved');
    } catch(e) { state.dirty=true; localDraft(); saveLabel('Not saved · draft retained'); throw e; }
    finally { state.saving=null; }
  })();
  await state.saving;
  if(state.dirty) await save();
}
async function refresh() { if(!state.current) return; const data=await api('/projects/'+state.current.project.id); if(!state.dirty&&!state.saving) state.current=data; else {state.current.images=data.images;state.current.jobs=data.jobs;} renderInspector(); }
function renderProjects() { $('#projects').innerHTML=state.projects.map(p=>`<button class="project-item ${state.current?.project.id===p.id?'active':''}" data-project="${p.id}">${h(p.name||'Untitled LoRA')}<small>${h(state.catalog.types.find(t=>t.id===p.kind)?.label)} · ${h(state.catalog.models.find(m=>m.id===p.model)?.label)}</small></button>`).join('')||'<p class="muted">No projects yet</p>'; }
async function selectProject(id) {
  await save();
  state.current=await api('/projects/'+id); state.recovered=false;
  let draft; try {draft=JSON.parse(localStorage.getItem(draftKey(id)));}catch{}
  if(draft) {state.recovered=draft;}
  localStorage.setItem('lora-studio.active',id); $('#project-title').textContent=state.current.project.name;
  renderProjects(); render(); saveLabel(state.recovered?'Unsaved draft available':'All changes saved');
}
function renderInspector() {
  if(!state.current){$('#inspector').innerHTML='<div class="card"><h3>From images to a LoRA</h3><p>Define what your LoRA should learn. Add images, prepare captions with a local vision model, then train with AI Toolkit.</p><p>Prompt Studio can supply the vision connection whenever it is running.</p></div>';return;}
  const {project:p,images,preflight}=state.current;
  const attrs=resolvedAttributes(p,state.catalog), ready=images.filter(i=>imageState(i,state.current.intent_digest).tone==='good').length;
  $('#inspector').innerHTML=`<div class="card"><span class="eyebrow">PROJECT AT A GLANCE</span><h3>${h(p.name||'Untitled LoRA')}</h3><div class="summary-row"><span>Base model</span><span>${h(state.catalog.models.find(m=>m.id===p.model).label)}</span></div><div class="summary-row"><span>Trigger</span><span>${h(p.trigger||'Not set')}</span></div><div class="summary-row"><span>Training images</span><span>${images.filter(i=>i.included).length}</span></div><div class="summary-row"><span>Captions ready</span><span>${ready}</span></div><div class="divider"></div><h3>Learning intent</h3><p><strong>Keep consistent</strong><br>${h(attrs.filter(a=>a.resolved==='keep').map(a=>a.label).join(', '))}</p><p><strong>Make changeable</strong><br>${h(attrs.filter(a=>a.resolved==='change').map(a=>a.label).join(', '))}</p>${p.also_keep?`<p>Also keep: ${h(p.also_keep)}</p>`:''}${p.also_change?`<p>Also change: ${h(p.also_change)}</p>`:''}</div><div class="card"><h3>A little variety goes a long way</h3><ul class="check-list"><li class="${p.description&&p.trigger?'done':''}">Define the target and trigger</li><li class="${images.length?'done':''}">Add clear, varied images</li><li class="${ready&&ready===images.filter(i=>i.included).length?'done':''}">Prepare and review captions</li><li class="${!preflight.errors.length?'done':''}">Pass the training checks</li></ul><p>Captions help separate the target from its context. Image variety is still essential.</p></div>`;
}
function field(label,key,value,placeholder='',extra='') {return `<label class="field">${label}<input data-field="${key}" value="${h(value)}" placeholder="${h(placeholder)}" ${extra}></label>`;}
function textarea(label,key,value,placeholder='',help='') {return `<label class="field">${label}<textarea data-field="${key}" placeholder="${h(placeholder)}">${h(value)}</textarea>${help?`<small>${help}</small>`:''}</label>`;}
function render() {
  $('.tabs').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.tab));
  renderInspector();
  if(!state.current) {$('#content').innerHTML='<div class="card empty"><div class="empty-icon">✧</div><h2>Teach a model something personal</h2><p>A character, an object, a visual style, or a concept. Start with what matters, and let your local vision model help with the captions.</p><button class="primary" data-action="new">Create your first LoRA</button></div>';return;}
  const content={setup:setupView,dataset:datasetView,training:trainingView,runs:runsView}[state.tab]();
  $('#content').innerHTML=(state.recovered?'<div class="note warning draft-banner">An unsaved draft is available from this browser. Choose which version to use before editing.<div class="actions"><button data-action="restore">Restore draft</button><button data-action="discard">Use saved version</button><button data-action="download-draft">Download draft</button></div></div>':'')+(state.recovered?'<fieldset class="draft-lock" disabled>'+content+'</fieldset>':content);
}
function setupView() {
  const p=state.current.project, attrs=state.catalog.types.find(t=>t.id===p.kind).attributes;
  const rows=items=>items.map(a=>{const choice=p.choices[a.id]||'auto';return `<div class="attribute-row"><div class="attr-name">${h(a.label)}<small>Auto: ${a.default==='keep'?'keep consistent':'make changeable'}</small></div><div class="segments" role="group" aria-label="${h(a.label)}">${['auto','keep','change'].map(c=>`<label><input type="radio" name="attr-${a.id}" data-attribute="${a.id}" value="${c}" ${choice===c?'checked':''}>${{auto:'Auto',keep:'Keep consistent',change:'Make changeable'}[c]}</label>`).join('')}</div></div>`}).join('');
  return `<section class="card"><div class="card-head"><div><h2>What are we teaching?</h2><p>Give your LoRA a name and define what makes it yours.</p></div><span class="badge">01 · SETUP</span></div><div class="form-grid">${field('LoRA name','name',p.name,'e.g. My ceramic collection','maxlength="120"')}${field('Trigger words','trigger',p.trigger,'e.g. crmc_vase','maxlength="120"')}<div class="full"><label class="field">LoRA type</label><div class="type-picker">${state.catalog.types.map(t=>`<button data-kind="${t.id}" class="${p.kind===t.id?'active':''}"><span>${icons[t.id]}</span>${h(t.label)}</button>`).join('')}</div></div><div class="full">${textarea('Describe the learning target','description',p.description,'What should the model learn? Identify the main subject or defining qualities.','Be specific about the target, especially if images contain several people or objects.')}</div><label class="field full">Base model<select data-field="model">${state.catalog.models.map(m=>`<option value="${m.id}" ${p.model===m.id?'selected':''}>${h(m.label)}</option>`).join('')}</select><small>The installed AI Toolkit handles model loading and training. Models may need to download on the first run.</small></label></div></section><section class="card"><div class="card-head"><div><h2>What stays. What can change.</h2><p>Choose what belongs to your LoRA’s identity and what you want to control in a prompt.</p></div></div><div class="attribute-head"><span>ATTRIBUTE</span><span>ONE CHOICE PER ATTRIBUTE</span></div>${rows(attrs.slice(0,5))}<details><summary>More attributes · ${attrs.length-5}</summary>${rows(attrs.slice(5))}</details><details><summary>Additional learning rules</summary><div class="form-grid extra-rules">${textarea('Also keep consistent','also_keep',p.also_keep,'One detail per line')}${textarea('Also make changeable','also_change',p.also_change,'One detail per line')}</div></details><div class="note">“Keep consistent” tells captioning to associate defining details with the trigger. “Make changeable” asks it to describe visible variation. These choices guide captions; they do not remove anything from the image.</div></section><div class="actions end"><button class="primary" data-goto="dataset">Continue to dataset →</button></div>`;
}
function datasetView() {
  const {images,intent_digest:digest}=state.current;
  return `<section class="card"><div class="card-head"><div><h2>Your training images</h2><p>Import images, then let your vision model prepare captions using your learning rules.</p></div><span class="badge">${images.filter(i=>i.included).length} INCLUDED</span></div><div class="drop-zone" id="drop-zone"><h3>Drop images here</h3><p>PNG, JPEG or WebP · up to 20 MB each · duplicates are skipped</p><div class="actions end"><button data-action="pick-images">Choose images</button><button class="quiet" data-action="pick-folder">Choose folder</button></div><input hidden id="image-input" type="file" accept="image/png,image/jpeg,image/webp" multiple><input hidden id="folder-input" type="file" accept="image/*" webkitdirectory multiple></div><div class="actions split"><span class="count">${images.length} images · ${images.filter(i=>imageState(i,digest).tone==='good').length} ready</span><button class="primary" data-action="caption" ${images.length?'':'disabled'}>✧ Prepare captions</button></div><p class="muted"><small>Missing and outdated automatic captions are prepared. Manual captions are preserved. Click any image to edit or review it.</small></p>${images.length?`<div class="image-grid">${images.map(i=>{const s=imageState(i,digest);return `<button class="image-card ${i.included?'':'excluded'}" data-image="${i.id}"><img src="/api/images/${i.id}" loading="lazy" alt="${h(i.name)}"><div class="image-info"><span class="filename">${h(i.name)}</span><span class="badge ${s.tone}">${s.label}</span><p>${h(i.caption||'Ready for your vision model.')}</p></div></button>`}).join('')}</div>`:'<div class="empty"><div class="empty-icon">▧</div><h3>No images yet</h3><p>Start with your own photos or existing generated images. Automatic dataset generation is not part of this first version.</p></div>'}</section><section class="card"><h3>Build a useful dataset</h3><p class="muted">Aim for variety in the attributes you want to change. Avoid near-duplicates and remove images with the wrong subject or obvious defects.</p><div class="coverage">${resolvedAttributes(state.current.project,state.catalog).filter(a=>a.resolved==='change').slice(0,6).map(a=>`<div>Vary ${h(a.label.toLowerCase())}</div>`).join('')}</div></section><div class="actions end"><button class="primary" data-goto="training">Review training →</button></div>`;
}
function trainingView() {
  const {project:p,preflight:c}=state.current;
  return `<section class="card"><div class="card-head"><div><h2>Ready when you are</h2><p>Start with conservative settings, or open the controls below to make them your own.</p></div><span class="badge">AI TOOLKIT</span></div><div class="form-grid"><div class="summary-row"><span>Training steps</span><span>${p.training.steps.toLocaleString()}</span></div><div class="summary-row"><span>LoRA rank</span><span>${p.training.rank}</span></div><div class="summary-row"><span>Resolution</span><span>${p.training.resolution} px</span></div><div class="summary-row"><span>Learning rate</span><span>${p.training.lr}</span></div></div><details><summary>Adjust training settings</summary><div class="form-grid extra-rules">${[['Steps','steps',100,20000,100],['LoRA rank','rank',4,128,1],['Resolution','resolution',256,1536,64],['Learning rate','lr',.000001,.01,.000001],['Seed','seed',0,2147483647,1],['GPU index','gpu',0,15,1]].map(([label,k,min,max,step])=>field(label,'training.'+k,p.training[k],'',`type="number" min="${min}" max="${max}" step="${step}"`)).join('')}<div class="full">${textarea('Preview prompts','sample_prompts',p.sample_prompts,'One prompt per line. Leave empty for automatic previews.','Use your trigger. Previews and checkpoints are saved every 250 steps (or at the final step for shorter runs).')}</div></div></details></section><section class="card"><h3>Before training</h3>${c.errors.length?`<div class="note warning"><strong>A few things need attention</strong><ul>${c.errors.map(e=>`<li>${h(e)}</li>`).join('')}</ul></div>`:'<div class="note"><span class="badge good">Ready</span> Dataset and backend checks passed.</div>'}${c.warnings.map(w=>`<p class="muted"><small>${h(w)}</small></p>`).join('')}<p class="muted">Each run gets its own copy of the images, captions and configuration. Later project edits won’t change that run.</p><div class="note">${state.settings.integration==='standalone'||!state.settings.unload_comfy?'Automatic ComfyUI unloading is disabled. Ensure the training GPU has enough free memory.':'When ComfyUI is available, Studio checks for queued work and requests a model unload before launch. Keep new generations paused while training.'}</div><div class="actions end extra-rules"><button data-action="export" ${c.errors.length?'disabled':''}>Export training bundle</button><button class="primary" data-action="train" ${c.errors.length?'disabled':''}>Start training →</button></div></section>`;
}
function runsView() {
  const jobs=[...state.current.jobs].sort((a,b)=>b.created-a.created);
  return `<section class="card"><div class="card-head"><div><h2>Runs & activity</h2><p>Caption jobs, immutable training bundles, logs and saved LoRAs.</p></div><span class="badge">${jobs.length} RUNS</span></div>${jobs.length?jobs.map(j=>{const active=['queued','running'].includes(j.status);return `<article class="run"><div class="run-head"><h3>${{caption:'Vision captioning',export:'Training bundle',train:'LoRA training'}[j.kind]}</h3><span class="badge ${j.status==='complete'?'good':j.status==='failed'?'bad':'warn'}">${h(j.status)}</span></div><small>${new Date(j.created*1000).toLocaleString()}${j.snapshot?' · '+h(j.snapshot.name):''}</small>${j.total?`<div class="progress"><progress max="${j.total}" value="${j.completed}"></progress></div><p>${j.completed} / ${j.total} images processed</p>`:''}${j.error?`<p class="danger">${h(j.error)}</p>`:''}${j.messages.map(m=>`<p>${h(m)}</p>`).join('')}<div class="actions"><button data-job="${j.id}">View details${j.bundle?' & files':''}</button>${active?`<button class="quiet danger" data-stop="${j.id}">Stop</button>`:''}</div></article>`}).join(''):'<div class="empty"><div class="empty-icon">↗</div><h3>Your activity will appear here</h3><p>Prepare captions or export a training bundle to start.</p></div>'}</section>`;
}
function modal(title,content) { $('#modal-content').innerHTML=`<div class="modal-head"><h2>${h(title)}</h2><button data-action="close-modal" aria-label="Close dialog">✕</button></div>${content}`; if(!$('#modal').open) $('#modal').showModal(); }
async function connections() {
  await save(); const s=state.settings;
  modal('Connections',`<form id="settings-form"><div class="form-grid"><label class="field full">Vision connection<select name="integration"><option value="auto">Automatic · use Prompt Studio when available</option><option value="promptstudio">Prompt Studio only</option><option value="standalone">Standalone local LLM</option></select><small>Automatic uses the active Prompt Studio chat’s saved LLM settings. This does not add messages to that chat.</small></label><label class="field full">ComfyUI / Prompt Studio URL<input name="prompt_studio_url" value="${h(s.prompt_studio_url)}"></label></div><label class="checkbox"><input name="unload_comfy" type="checkbox" ${s.unload_comfy?'checked':''}>Request ComfyUI model unload before training when idle</label><details open><summary>Standalone endpoint / automatic fallback</summary><div class="form-grid"><label class="field">OpenAI-compatible base URL<input name="llm_url" value="${h(s.llm_url)}"></label><label class="field">Vision model ID<input name="llm_model" value="${h(s.llm_model)}" placeholder="Exact model ID from your local server"></label></div><p class="muted"><small>Local HTTP endpoints only. The selected model must accept images.</small></p></details><div class="divider"></div><h3>AI Toolkit installation</h3><p class="muted">Studio calls your existing installation. Update AI Toolkit independently.</p><div class="form-grid"><label class="field full">AI Toolkit folder<input name="toolkit_path" value="${h(s.toolkit_path)}" placeholder="Folder containing run.py"></label><label class="field full">AI Toolkit Python executable<input name="toolkit_python" value="${h(s.toolkit_python)}" placeholder="Full path to python.exe"></label></div><div id="connection-result" class="note">${h(connectionText())}</div><div class="actions end extra-rules"><button type="submit" class="primary">Save & check connections</button></div></form>`);
  $('#settings-form [name=integration]').value=s.integration;
}
function connectionText() {
  if(!state.connection) return 'Connections have not been checked yet.';
  const c=state.connection, ps=c.promptstudio;
  return (ps.available?`Prompt Studio · ${ps.provider} · ${ps.vision?.available?'vision ready':'vision unavailable: '+(ps.vision?.reason||'select a vision model')}`:c.standalone.available?'Standalone endpoint available'+(state.settings.llm_model?'':' · choose a model ID'):'Vision connection unavailable')+' · AI Toolkit '+(c.backend.available?'ready':'not configured');
}
async function checkConnections() {
  state.connection=await api('/connection'); $('#connection-label').textContent=connectionText(); if($('#connection-result')) $('#connection-result').textContent=connectionText();
}
function editImage(id) {
  const i=state.current.images.find(i=>i.id===id), status=imageState(i,state.current.intent_digest);
  modal('Review caption',`<form id="image-form" data-id="${i.id}" data-revision="${i.revision}"><div class="modal-grid"><div><img src="/api/images/${i.id}" alt="${h(i.name)}"><p class="muted">${h(i.name)}</p><span class="badge ${status.tone}">${status.label}</span><label class="checkbox"><input name="included" type="checkbox" ${i.included?'checked':''}>Include in training</label></div><div><label class="field">Training caption<textarea name="caption" maxlength="6000">${h(i.caption)}</textarea><small>Manual edits are preserved during automatic captioning.</small></label><label class="checkbox"><input name="reviewed" type="checkbox">I reviewed this caption against the current learning rules</label>${i.warnings.length?`<div class="note warning">${i.warnings.map(w=>`<p>${h(w)}</p>`).join('')}</div>`:''}<details><summary>Original visual observations</summary><p>${h(i.observations||'No vision analysis yet.')}</p></details><details><summary>Caption history · ${i.history.length}</summary>${[...i.history].reverse().map(v=>`<p><small>${new Date(v.at*1000).toLocaleString()} · ${h(v.source)}</small><br>${h(v.caption||'(Empty)')}</p>`).join('')}</details></div></div><div class="actions end extra-rules"><button type="submit" class="primary">Save caption</button></div></form>`);
}
async function jobDetails(id) {
  const {job:j,files}=await api('/jobs/'+id);
  modal('Run details',`<p><span class="badge">${h(j.status)}</span> ${h(j.kind)} · ${new Date(j.created*1000).toLocaleString()}</p>${j.error?`<div class="note error">${h(j.error)}</div>`:''}${j.messages.map(m=>`<p>${h(m)}</p>`).join('')}<div class="file-list">${files.map(f=>`<a href="/api/jobs/${id}/file?path=${encodeURIComponent(f.path)}" target="_blank" rel="noopener">${h(f.path)} <small>(${Math.ceil(f.size/1024)} KB)</small></a>`).join('')||'<p class="muted">No exported files yet.</p>'}</div>${files.some(f=>f.path==='training.log')?'<pre id="log-preview">Loading latest log…</pre>':''}<div class="actions end extra-rules">${j.status==='interrupted'&&j.kind==='train'?`<button data-recover="${id}">Check previous trainer process</button>`:''}<button data-job="${id}">Refresh details</button></div>`);
  if($('#log-preview')) {const log=await fetch(`/api/jobs/${id}/file?path=training.log`).then(r=>r.text());if($('#log-preview'))$('#log-preview').textContent=log.slice(-18000)||'Waiting for trainer output…';}
}
async function importImages(files) {
  await save(); const pid=state.current.project.id; let count=0,duplicates=0; const errors=[];
  for(const file of files) {
    if(!/^image\/(png|jpeg|webp)$/.test(file.type)) continue;
    try {
      if(file.size>20*1024*1024) throw new Error('larger than 20 MB');
      const bitmap=await createImageBitmap(file); if(bitmap.width*bitmap.height>100000000) {bitmap.close();throw new Error('image dimensions are too large');}
      let upload=file;
      if(file.type==='image/webp') {const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;canvas.getContext('2d').drawImage(bitmap,0,0);upload=await new Promise(r=>canvas.toBlob(r,'image/png'));}
      bitmap.close();
      const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(upload);});
      const result=await api(`/projects/${pid}/images`,'POST',{name:file.name,data});result.duplicate?duplicates++:count++;
      saveLabel(`Imported ${count} images…`);
    } catch(e) {errors.push(`${file.name}: ${e.message}`);}
  }
  await refresh();render();saveLabel('All changes saved');toast(`Imported ${count} images${duplicates?` · ${duplicates} duplicates skipped`:''}${errors.length?' · '+errors.join('; '):''}`,errors.length>0);
}
async function action(name) {
  if(name==='new') {await save();const p=await api('/projects','POST',{});state.projects.unshift(p);state.tab='setup';await selectProject(p.id);return;}
  if(name==='close-modal') {$('#modal').close();return;}
  if(name==='pick-images'||name==='pick-folder') {$(name==='pick-images'?'#image-input':'#folder-input').click();return;}
  if(name==='restore') {const revision=state.current.project.revision;state.current.project={...state.recovered,revision};state.recovered=false;changed();render();return;}
  if(name==='discard') {localStorage.removeItem(draftKey(state.current.project.id));state.recovered=false;render();return;}
  if(name==='download-draft') {const blob=new Blob([JSON.stringify(state.recovered,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='lora-project-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);return;}
  if(['caption','export','train'].includes(name)) {
    await save();if(state.recovered)throw new Error('Restore or discard the unsaved draft before starting a job.');
    await api(`/projects/${state.current.project.id}/${name}`,'POST',{});state.tab='runs';await refresh();render();toast(name==='train'?'Training launch queued. Watch the log in Runs.':'Job started.');
  }
}
document.addEventListener('click',async e=>{
  const b=e.target.closest('button');if(!b)return;
  try {
    if(b.id==='new-project') return await action('new');
    if(b.id==='connections') return await connections();
    if(b.dataset.project) return await selectProject(b.dataset.project);
    if(b.dataset.tab||b.dataset.goto) {await save();state.tab=b.dataset.tab||b.dataset.goto;await refresh();render();return;}
    if(b.dataset.kind) {state.current.project.kind=b.dataset.kind;changed();render();return;}
    if(b.dataset.action) return await action(b.dataset.action);
    if(b.dataset.image) return editImage(b.dataset.image);
    if(b.dataset.job) return await jobDetails(b.dataset.job);
    if(b.dataset.recover) {await api(`/jobs/${b.dataset.recover}/recover`,'POST',{});await refresh();render();return await jobDetails(b.dataset.recover);}
    if(b.dataset.stop) {const j=state.current.jobs.find(j=>j.id===b.dataset.stop);modal('Stop this job?',`<p>${j.kind==='caption'?'Stops after the current vision request returns. Completed captions stay saved.':'The trainer will stop. Work since the last checkpoint may be lost.'}</p><div class="actions end"><button data-action="close-modal">Keep running</button><button class="danger" data-confirm-stop="${j.id}">Stop job</button></div>`);return;}
    if(b.dataset.confirmStop) {await api(`/jobs/${b.dataset.confirmStop}/stop`,'POST',{});$('#modal').close();toast('Stop requested.');}
  }catch(err){toast(err.message,true);}
});
document.addEventListener('input',e=>{
  const key=e.target.dataset.field;if(!key)return;
  if(key.startsWith('training.'))state.current.project.training[key.split('.')[1]]=Number(e.target.value);else state.current.project[key]=e.target.value;
  changed();
});
document.addEventListener('change',async e=>{
  if(e.target.dataset.attribute){state.current.project.choices[e.target.dataset.attribute]=e.target.value;changed();}
  if(e.target.id==='image-input'||e.target.id==='folder-input'){try{await importImages([...e.target.files]);}catch(err){toast(err.message,true);}}
});
document.addEventListener('submit',async e=>{
  e.preventDefault();const form=e.target,button=form.querySelector('[type=submit]');button.disabled=true;
  try{
    if(form.id==='settings-form'){const data=Object.fromEntries(new FormData(form));data.unload_comfy=form.elements.unload_comfy.checked;state.settings=await api('/settings','PUT',data);await checkConnections();await refresh();render();toast('Connections saved.');}
    if(form.id==='image-form'){await api('/images/'+form.dataset.id,'PUT',{revision:Number(form.dataset.revision),caption:form.elements.caption.value,included:form.elements.included.checked,reviewed:form.elements.reviewed.checked});$('#modal').close();await refresh();render();toast('Caption saved.');}
  }catch(err){toast(err.message,true);}finally{button.disabled=false;}
});
document.addEventListener('dragover',e=>{const zone=e.target.closest('#drop-zone');if(zone){e.preventDefault();zone.classList.add('drag');}});
document.addEventListener('dragleave',e=>{e.target.closest('#drop-zone')?.classList.remove('drag');});
document.addEventListener('drop',async e=>{if(e.target.closest('#drop-zone')){e.preventDefault();try{await importImages([...e.dataTransfer.files]);}catch(err){toast(err.message,true);}}});
window.addEventListener('beforeunload',e=>{if(state.dirty||state.saving){e.preventDefault();e.returnValue='';}});
async function boot(){const b=await api('/bootstrap');Object.assign(state,b);renderProjects();render();const active=localStorage.getItem('lora-studio.active');if(state.projects.length)await selectProject(state.projects.some(p=>p.id===active)?active:state.projects[0].id);checkConnections().catch(e=>{$('#connection-label').textContent=e.message;});}
boot().catch(e=>{toast(e.message,true);saveLabel('Could not load workspace');});
setInterval(async()=>{if(!state.current||state.dirty||state.saving||$('#modal').open)return;const active=state.current.jobs.some(j=>['queued','running'].includes(j.status));if(!active)return;try{await refresh();if(['runs','dataset','training'].includes(state.tab))render();}catch{}},2500);
