import './image-preview.js';
import {resultsView,completedRun} from './results.js';
import {checkpointComparison} from './comparisons.js';
import {preparationLocked,lockNotice,disablePreparation,enablePreparation} from './project-lifecycle.js';
import {setupWizard} from './setup.js';
import {conceptSettings} from './concept-preparation.js';
import {styleSettings} from './style-preparation.js';
import {readObjectForm,restoreObjectForm,bindObjectEditor,drawObjectMask} from './object-preparation.js';
import {metadataPanel,setMetadataField} from './lora-metadata.js';
import {presetPanel,previewPanel,promptMode} from './training.js';
import {trainingLabel,visibleTraining,activityCards,projectTrainingBadge} from './activity.js';
import {escapeHTML as h, imageState, resolvedAttributes} from './domain.js';
import {captionSettings,updateCaptionSettings,captionSourceLabel} from './captioning.js';
import {prepSettings,prepSummary,imagePrepInfo,reviewPreparation,drawCrop,coverageSummary,coverageControls} from './preparation.js';

import {workflowControl,reviewPanel,filterImages,captionEvidence} from './review.js';

const $ = s => document.querySelector(s);
let datasetFilter='all', trainingJobs=[], activityPending=false, activityEndpoint=true, dashboardJob=null;
const trainingProgress=new Map();
const navigation=new URLSearchParams(location.search);
let resultsJob=null, resultsData=null, resultsError='', resultsRequest=0;
let deletingProject=null;
const state = {projects:[], catalog:null, settings:null, current:null, tab:'setup', dirty:false, saving:null, timer:null, connection:null, recovered:false};
const comparisons=checkpointComparison({api,modal,toast,refreshResults:async()=>{resultsData=null;await loadResults(resultsJob);render();}});
const openSetup = setupWizard({api, state, toast, onSaved:()=>{checkConnections().catch(()=>{});refresh().then(render).catch(()=>{});}});
const icons = {character:'◎',object:'◇',style:'✧',concept:'↔'};
const draftKey = id => `lora-studio.draft.${id}`;
let toastTimer;
let connectionCheck, connectionError='', connectionChecked=0;
function toast(message, error=false) { const el=$('#toast'); el.textContent=message; el.className=error?'error':''; el.hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true,error?12000:4500); }
async function api(path, method='GET', value) { const response=await fetch('/api'+path,{method,headers:value?{'Content-Type':'application/json'}:{},body:value?JSON.stringify(value):undefined}); const data=await response.json(); if(!response.ok){const error=new Error(data.error||'Request failed');error.status=response.status;error.code=data.code;throw error;} return data; }
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
    } catch(e) { state.dirty=true; localDraft(); saveLabel('Not saved · draft retained'); if(e.code==='preparation_locked'){state.recovered=structuredClone(state.current.project);state.dirty=false;state.current=await api('/projects/'+snapshot.id);render();} throw e; }
    finally { state.saving=null; }
  })();
  await state.saving;
  if(state.dirty) await save();
}
async function refresh() {
  if(!state.current||deletingProject===state.current.project.id)return;
  const projectId=state.current.project.id, previous=state.current.jobs;
  const data=await api('/projects/'+projectId);if(state.current?.project.id!==projectId||deletingProject===projectId)return;
  const finished=completedRun(previous,data.jobs,state.tab,dashboardJob);
  if(finished){resultsJob=finished;resultsData=null;state.tab='results';}
  if(preparationLocked(data.project)&&state.dirty&&!state.saving){
    localDraft();state.recovered=structuredClone(state.current.project);state.dirty=false;clearTimeout(state.timer);
    saveLabel('Preparation locked · unsaved draft retained');
  }
  if(!state.dirty&&!state.saving)state.current=data;
  else {state.current.images=data.images;state.current.jobs=data.jobs;}
  rememberJobs(data.project.id,data.jobs);if(state.tab==='training')await loadPreviewSuite();renderProjects();renderInspector();renderPreparationLock();
}
function renderPreparationLock(){
  const project=state.current?.project, notice=lockNotice(project), container=$('#preparation-status');
  if(container.innerHTML!==notice)container.innerHTML=notice;
  container.hidden=!notice;
  const dropZone=$('#drop-zone');
  if(dropZone)dropZone.hidden=preparationLocked(project);
  if(!preparationLocked(project)){enablePreparation($('#content'));enablePreparation($('#image-form'));enablePreparation($('#image-draft-banner'));return;}
  if(['setup','dataset','training'].includes(state.tab))disablePreparation($('#content'));
  if($('#image-form')){disablePreparation($('#image-form'));disablePreparation($('#image-draft-banner'));}
}
function rememberJobs(project,jobs){trainingJobs=[...trainingJobs.filter(j=>j.project!==project),...jobs.filter(j=>j.kind==='train')];}
async function refreshActivity(){if(activityPending)return;activityPending=true;try{
  if(activityEndpoint){try{trainingJobs=(await api('/activity')).jobs;}catch(e){if(e.status!==404)throw e;activityEndpoint=false;}}
  // Older running servers can serve the updated UI without interrupting their trainer.
  if(!activityEndpoint){const results=await Promise.allSettled(state.projects.map(p=>api('/projects/'+p.id)));for(const result of results)if(result.status==='fulfilled')rememberJobs(result.value.project.id,result.value.jobs);}
  renderProjects();
  const running=trainingJobs.filter(j=>j.status==='running'&&j.pid&&j.phase==='Training');
  for(const id of trainingProgress.keys())if(!running.some(j=>j.id===id))trainingProgress.delete(id);
  await Promise.allSettled(running.map(async job=>{
    const {step,total}=await api('/jobs/'+encodeURIComponent(job.id)+'/monitor');
    trainingProgress.set(job.id,{step,total});
  }));
  renderProjects();
}catch{}finally{activityPending=false;}}
function renderProjects() {
  $('#training-activity').innerHTML=activityCards(trainingJobs.filter(j=>state.projects.some(p=>p.id===j.project)),state.projects,trainingProgress);
  $('#projects').innerHTML=state.projects.map(p=>`<div class="project-row ${state.current?.project.id===p.id?'active':''}"><button class="project-item" data-project="${p.id}"><span class="project-name">${h(p.name||'Untitled LoRA')}</span><small>${h(state.catalog.types.find(t=>t.id===p.kind)?.label)} · ${h(state.catalog.models.find(m=>m.id===p.model)?.label)}</small>${projectTrainingBadge(trainingJobs,p.id)}</button><button class="project-delete quiet danger" data-delete-project="${p.id}" aria-label="Delete session ${h(p.name||'Untitled LoRA')}">Delete</button></div>`).join('')||'<p class="muted">No projects yet</p>';
}
async function confirmProjectDeletion(id) {
  await save();
  const {project}=await api('/projects/'+id);
  modal('Delete session?',`<p>Delete <strong>${h(project.name||'Untitled LoRA')}</strong>?</p><p><strong>Delete session only</strong> removes it from Studio and keeps its associated files on disk.</p><p><strong>Delete session and files</strong> also permanently removes Studio’s imported image copies, captions, analysis files, run exports, checkpoints, previews and saved comparisons for this session.</p><p>Original source images and shared models are kept. This cannot be undone.</p><div class="actions end"><button data-action="close-modal" autofocus>Cancel</button><button data-confirm-delete-project="${id}" data-delete-revision="${project.revision}" data-delete-files="false">Delete session only</button><button class="danger" data-confirm-delete-project="${id}" data-delete-revision="${project.revision}" data-delete-files="true">Delete session and files</button></div>`);
  $('#modal [autofocus]').focus();
}
async function deleteProject(button) {
  const id=button.dataset.confirmDeleteProject, dialog=$('#modal');
  const controls=[...dialog.querySelectorAll('button')];
  const preventCancel=e=>e.preventDefault();
  controls.forEach(b=>b.disabled=true);dialog.addEventListener('cancel',preventCancel);
  deletingProject=id;
  try {
    const result=await api('/projects/'+id+'/delete','POST',{revision:Number(button.dataset.deleteRevision),delete_files:button.dataset.deleteFiles==='true'});
    state.projects=state.projects.filter(p=>p.id!==id);
    trainingJobs=trainingJobs.filter(j=>j.project!==id);
    try {localStorage.removeItem(draftKey(id));for(const image of result.image_ids)localStorage.removeItem(imageDraftKey(image));if(localStorage.getItem('lora-studio.active')===id)localStorage.removeItem('lora-studio.active');}catch{}
    if(state.current?.project.id===id){
      clearTimeout(state.timer);comparisons.leave();resultsRequest++;
      Object.assign(state,{current:null,dirty:false,recovered:false,previewSuite:null,tab:'setup'});
      dashboardJob=null;resultsJob=null;resultsData=null;resultsError='';
      $('#project-title').textContent='Your LoRA workspace';saveLabel('');
      history.replaceState(null,'',location.pathname);render();
    }
    renderProjects();dialog.close();
    if(result.cleanup_pending)toast('Session deleted. Some files could not be removed and remain in Studio’s data/session-trash folder.',true);
    else toast(result.delete_files?'Session and associated files deleted.':'Session deleted. Associated files kept on disk.');
    if(!state.current&&state.projects.length)await selectProject(state.projects[0].id);
  } finally {deletingProject=null;controls.forEach(b=>b.disabled=false);dialog.removeEventListener('cancel',preventCancel);}
}
function updateNavigation(){if(!state.current)return;const params=new URLSearchParams({project:state.current.project.id,tab:state.tab});if(state.tab==='dashboard'&&dashboardJob)params.set('job',dashboardJob);if(state.tab==='results'&&resultsJob)params.set('job',resultsJob);history.replaceState(null,'','?'+params);}
function dashboardView(){
  const jobs=state.current.jobs.filter(j=>j.kind==='train').sort((a,b)=>b.created-a.created);
  const job=jobs.find(j=>j.id===dashboardJob)||visibleTraining(jobs)[0]||jobs[0];
  if(!job)return '<section class="card empty"><h2>Training dashboard</h2><p>Start training to see live loss, checkpoints and test generations here.</p><button data-goto="training">Go to Create →</button></section>';
  dashboardJob=job.id;updateNavigation();
  return `<div class="dashboard-tools"><button data-results-job="${job.id}">View results</button><button class="quiet" data-goto="runs">Runs & activity</button></div><iframe class="training-dashboard-frame" title="Training dashboard" src="/monitor.html?embedded=1&job=${encodeURIComponent(job.id)}"></iframe>`;
}
async function loadPreviewSuite(){const result=await Promise.allSettled([api('/projects/'+state.current.project.id+'/sample-prompts'),api('/presets')]);state.previewSuite=result[0].status==='fulfilled'?result[0].value:null;if(result[1].status==='fulfilled')state.presets=result[1].value.presets;}
async function selectProject(id) {
  comparisons.leave();
  await save();
  resultsRequest++;resultsData=null;resultsError='';
  state.current=await api('/projects/'+id);rememberJobs(id,state.current.jobs); state.recovered=false;
  let draft; try {draft=JSON.parse(localStorage.getItem(draftKey(id)));}catch{}
  if(draft) {state.recovered=draft;}
  state.previewSuite=null;if(state.tab==='training')await loadPreviewSuite();localStorage.setItem('lora-studio.active',id); $('#project-title').textContent=state.current.project.name;
  renderProjects(); render(); saveLabel(state.recovered?'Unsaved draft available':'All changes saved');
}
function renderInspector() {
  if(!state.current){$('#inspector').innerHTML='<div class="card"><h3>From images to a LoRA</h3><p>Define what your LoRA should learn. Add images, prepare captions with a local vision model, then train with AI Toolkit.</p><p>Prompt Studio can supply the vision connection whenever it is running.</p></div>';return;}
  const {project:p,images,preflight}=state.current;
  const attrs=resolvedAttributes(p,state.catalog), ready=images.filter(i=>imageState(i,state.current.intent_digest).tone==='good').length;
  $('#inspector').innerHTML=`<div class="card"><span class="eyebrow">PROJECT AT A GLANCE</span><h3>${h(p.name||'Untitled LoRA')}</h3><div class="summary-row"><span>Base model</span><span>${h(state.catalog.models.find(m=>m.id===p.model).label)}</span></div><div class="summary-row"><span>Trigger</span><span>${h(p.trigger||'Not set')}</span></div><div class="summary-row"><span>Training images</span><span>${images.filter(i=>i.included).length}</span></div><div class="summary-row"><span>Captions ready</span><span>${ready}</span></div><div class="divider"></div><h3>Learning intent</h3><p><strong>Keep consistent</strong><br>${h(attrs.filter(a=>a.resolved==='keep').map(a=>a.label).join(', '))}</p><p><strong>Make changeable</strong><br>${h(attrs.filter(a=>a.resolved==='change').map(a=>a.label).join(', '))}</p>${p.also_keep?`<p>Also keep: ${h(p.also_keep)}</p>`:''}${p.also_change?`<p>Also change: ${h(p.also_change)}</p>`:''}</div><div class="card"><h3>A little variety goes a long way</h3><ul class="check-list"><li class="${p.trigger?'done':''}">Choose the LoRA type and trigger</li><li class="${images.length?'done':''}">Add clear, varied images</li><li class="${ready&&ready===images.filter(i=>i.included).length?'done':''}">Prepare and review captions</li><li class="${!preflight.errors.length?'done':''}">Pass the training checks</li></ul><p>Captions help separate the target from its context. Image variety is still essential.</p></div>`;
}
function field(label,key,value,placeholder='',extra='') {return `<label class="field">${label}<input data-field="${key}" value="${h(value)}" placeholder="${h(placeholder)}" ${extra}></label>`;}
function textarea(label,key,value,placeholder='',help='') {return `<label class="field">${label}<textarea data-field="${key}" placeholder="${h(placeholder)}">${h(value)}</textarea>${help?`<small>${help}</small>`:''}</label>`;}
function render() {
  const openOptions=[...document.querySelectorAll('.optional-settings[open][id]')].map(el=>el.id);
  updateNavigation();$('.workspace').classList.toggle('dashboard-workspace',state.tab==='dashboard');$('.workspace').classList.add('simple-workspace');
  $('.tabs').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.tab));
  renderInspector();
  renderPreparationLock();
  if(!state.current) {$('#content').innerHTML='<div class="card empty"><div class="empty-icon">✧</div><h2>Teach a model something personal</h2><p>A character, an object, a visual style, or a concept. Start with what matters, and let your local vision model help with the captions.</p><button class="primary" data-action="new">Create your first LoRA</button></div>';return;}
  if(state.tab==='results'){renderResults();return;}
  if(state.tab==='dashboard'){$('#content').innerHTML=dashboardView();return;}
  const content={setup:setupView,dataset:datasetView,training:trainingView,runs:runsView}[state.tab]();
  $('#content').innerHTML=(state.recovered?'<div class="note warning draft-banner">An unsaved draft is available from this browser. '+(preparationLocked(state.current.project)?'Locked preparation cannot be replaced; download the draft to keep it.':'Choose which version to use before editing.')+'<div class="actions"><button data-action="restore">Restore draft</button><button data-action="discard">Use saved version</button><button data-action="download-draft">Download draft</button></div></div>':'')+(state.recovered&&!preparationLocked(state.current.project)?'<fieldset class="draft-lock" disabled>'+content+'</fieldset>':content);
  for(const id of openOptions){const el=document.getElementById(id);if(el)el.open=true;}
  renderPreparationLock();
}
function setupView() {
  const p=state.current.project, attrs=state.catalog.types.find(t=>t.id===p.kind).attributes;
  const rows=items=>items.map(a=>{const choice=p.choices[a.id]||'auto';return `<div class="attribute-row"><div class="attr-name">${h(a.label)}<small>Auto: ${a.default==='keep'?'keep consistent':'make changeable'}</small></div><div class="segments" role="group" aria-label="${h(a.label)}">${['auto','keep','change'].map(c=>`<label><input type="radio" name="attr-${a.id}" data-attribute="${a.id}" value="${c}" ${choice===c?'checked':''}>${{auto:'Auto',keep:'Keep consistent',change:'Make changeable'}[c]}</label>`).join('')}</div></div>`}).join('');
  return `<section class="card"><div class="card-head"><div><h2>What are we teaching?</h2><p>Give your LoRA a name and define what makes it yours.</p></div><span class="badge">01 · BASE INFO</span></div><div class="form-grid">${field('LoRA name','name',p.name,'e.g. My ceramic collection','maxlength="120"')}<div>${field('Trigger words','trigger',p.trigger,'e.g. crmc_vase','maxlength="120"')}${state.catalog.types.find(t=>t.id===p.kind).trigger_guidance?.[p.model]?`<p class="muted">${h(state.catalog.types.find(t=>t.id===p.kind).trigger_guidance[p.model])}</p>`:''}<small class="muted">Generated for new projects; edit it here.</small></div><div class="full"><label class="field">LoRA type</label><div class="type-picker">${state.catalog.types.map(t=>`<button data-kind="${t.id}" class="${p.kind===t.id?'active':''}"><span>${icons[t.id]}</span>${h(t.label)}</button>`).join('')}</div></div><div class="full">${textarea('Describe the learning target · optional','description',p.description,state.catalog.types.find(t=>t.id===p.kind).default_description,'Leave empty to use the default for your LoRA type, with your Keep/Change rules. Add details to distinguish a target or refine what to learn.')}</div>${styleSettings(p,state.catalog)}${conceptSettings(p,state.catalog)}<label class="field full">Base model<select data-field="model">${state.catalog.models.map(m=>`<option value="${m.id}" ${p.model===m.id?'selected':''}>${h(m.label)}</option>`).join('')}</select><small>The installed AI Toolkit handles model loading and training. Models may need to download on the first run.</small></label></div></section><details class="card optional-settings" id="learning-options"><summary>Optional · learning rules</summary><p class="muted">Defaults follow your LoRA type. Adjust only what matters to you.</p><h3 class="extra-rules">What stays. What can change.</h3><div class="attribute-head"><span>ATTRIBUTE</span><span>ONE CHOICE PER ATTRIBUTE</span></div>${rows(attrs)}<details><summary>Additional learning rules</summary><div class="form-grid extra-rules">${textarea('Also keep consistent','also_keep',p.also_keep,'One detail per line')}${textarea('Also make changeable','also_change',p.also_change,'One detail per line')}</div></details><p class="muted"><small>Keep/change rules guide captions. They do not remove image content or guarantee disentanglement.</small></p></details><div class="actions end"><button class="primary" data-goto="dataset">Continue to dataset →</button></div>`;
}
function datasetView() {
  const {images,intent_digest:digest}=state.current;
  const locked=preparationLocked(state.current.project);
  return `<section class="card"><div class="card-head"><div><h2>Your training images</h2><p>${locked?'Review the images saved for this run. Use Copy to new run to change the dataset.':'Add your images, then continue to Create. Captions and image preparation happen automatically.'}</p></div><span class="badge">${images.filter(i=>i.included).length} INCLUDED</span></div>${locked?'':`<div class="drop-zone" id="drop-zone"><h3>Drop images here</h3><p>PNG, JPEG or WebP · up to 20 MB each · duplicates are skipped</p><div class="actions end"><button data-action="pick-images">Choose images</button><button class="quiet" data-action="pick-folder">Choose folder</button></div><input hidden id="image-input" type="file" accept="image/png,image/jpeg,image/webp" multiple><input hidden id="folder-input" type="file" accept="image/*" webkitdirectory multiple></div>`}<div class="actions split"><span class="count">${images.length} images · ${images.filter(i=>imageState(i,digest).tone==='good').length} ready</span><button class="primary" data-goto="training" ${images.some(i=>i.included)?'':'disabled'}>Continue to Create →</button></div><p class="muted"><small>Click an image to open its dataset entry and ${locked?'review its caption and findings.':'review its caption or override a finding.'} Enlarge images from inside the entry.</small></p>${images.length?`<label class="field">Show images<select id="dataset-filter">${[["all","All images"],["concerns","Needs attention"],["drafts","Unreviewed drafts"],["excluded","Excluded"]].map(([v,l])=>`<option value="${v}" ${datasetFilter===v?"selected":""}>${l}</option>`).join("")}</select></label><div class="image-grid">${filterImages(images,datasetFilter).map(i=>{const s=imageState(i,digest);return `<article class="image-card ${i.included?'':'excluded'}" ><button type="button" class="image-preview-trigger" data-image="${i.id}" aria-label="Review ${h(i.name)}"><img src="/api/images/${i.id}" loading="lazy" alt="${h(i.name)}"></button><button type="button" class="image-info" data-image="${i.id}"><span class="image-review-label">Review image</span><span class="filename">${h(i.name)}</span><span class="badge ${s.tone}">${s.label}</span>${imagePrepInfo(i)}<p>${h(i.caption||'Ready for your vision model.')}</p></button></article>`}).join('')}</div>`:'<div class="empty"><div class="empty-icon">▧</div><h3>No images yet</h3><p>Start with your own photos or existing generated images. Automatic dataset generation is not part of this first version.</p></div>'}</section><details class="card optional-settings" id="dataset-options"><summary>Optional · preparation, captions & checks</summary>${prepSettings(state.current.project,state.catalog)}${prepSummary(images,state.current.project,state.catalog)}${coverageSummary(images,state.current.project,state.catalog)}<p class="muted">Caption source: <strong>${h(captionSourceLabel(state.settings.caption_source))}</strong> <button class="quiet" data-action="settings">Change</button></p><div class="actions"><button data-action="prepare" ${images.length?'':'disabled'}>Prepare dataset only</button><button data-action="caption" ${images.length?'':'disabled'}>Prepare captions</button></div><details><summary>Reprocess caption text</summary><p class="muted">Reuse saved JoyCaption, custom-model, Style or Concept observations with the current learning rules. Original observations and prior captions are retained. Manual captions are skipped; changed crops need fresh visual observations.</p><button data-action="reprocess-captions" ${images.some(i=>i.included&&!i.manual&&i.observations&&(['joycaption','custom'].includes(i.caption_source)||['style','concept'].includes(state.current.project.kind)&&i.observation_kind===state.current.project.kind))?'':'disabled'}>Reprocess saved observations</button></details><p class="muted">Use varied images of your target. Missing and outdated automatic captions are prepared as needed; manual captions and overrides stay yours.</p></details>`;
}
function trainingView() {
  const {project:p}=state.current; const c=p.training_mode==='automatic'?state.current.automatic_preflight:state.current.preflight; const profile=state.catalog.models.find(m=>m.id===p.model).preparation;
  const count=state.current.images.filter(i=>i.included).length, locked=preparationLocked(p);
  return `<section class="card create-card"><div class="card-head"><div><span class="eyebrow">03 · CREATE</span><h2>${locked?'Saved training preparation':'Create your LoRA'}</h2><p>${locked?'Review the inputs saved for this run. Use Copy to new run to make changes.':p.training_mode==='automatic'?'We’ll prepare the dataset, generate preview prompts and train with your saved settings.':'Review-first mode is selected. Prepare your dataset before starting training.'}</p></div></div><div class="create-summary"><span>${h(state.catalog.models.find(m=>m.id===p.model).label)}</span><span>${count} images</span><span>${p.training.steps.toLocaleString()} steps</span><span>${p.training.resolution} px</span></div><p class="muted">${locked?'These settings are fixed for this run.':(p.training_preset?'Saved preset settings are applied.':'Your saved settings are ready to use.')+' You can customize everything below.'}</p>${!locked&&c.errors.length?`<div class="note warning"><strong>Before you can create</strong><ul>${c.errors.map(e=>`<li>${h(e)}</li>`).join('')}</ul><div class="actions"><button data-goto="setup">Base info</button><button data-goto="dataset">Dataset</button><button data-action="settings">Connections</button></div></div>`:''}<div class="actions end extra-rules"><button class="primary create-button" data-action="train" ${locked||c.errors.length?'disabled':''}>${locked?'Preparation locked':'Create LoRA →'}</button></div><p class="muted create-hint"><small>${locked?'Follow this run in Dashboard or Results.':'Starts training. Follow progress, checkpoints and test generations in Dashboard.'}</small></p></section>${metadataPanel(p)}<details class="card optional-settings" id="create-options"><summary>Optional · presets, preview prompts & training settings</summary>${presetPanel(p,state.presets)}${previewPanel(p,state.previewSuite,state.current.jobs.some(j=>j.kind==='preview_prompts'&&['queued','running'].includes(j.status)))}${workflowControl(p)}<details><summary>Adjust training settings</summary><div class="form-grid extra-rules">${[['Steps','steps',100,20000,100],['LoRA rank','rank',4,128,1],['Resolution','resolution',256,profile.maximum,profile.multiple],['Learning rate','lr',.000001,.01,.000001],['Seed','seed',0,2147483647,1],['GPU index','gpu',0,15,1],['Checkpoint interval · steps','checkpoint_every',1,20000,1],['Preview interval · steps (0 disables)','sample_every',0,20000,1],['Checkpoints to retain','keep_checkpoints',1,1000,1],['Preview denoising steps','sample_steps',1,200,1]].map(([label,k,min,max,step])=>field(label,'training.'+k,p.training[k]??({checkpoint_every:250,sample_every:250,keep_checkpoints:8,sample_steps:p.model==='qwen21'?40:30}[k]),'',`type="number" min="${min}" max="${max}" step="${step}"`)).join('')}${p.model==="krea2"?`<label class="field">Krea Raw weights<select data-field="training.krea_weights">${[["original","Original Krea release (requires access)"],["comfy_bf16","Comfy-Org BF16 repack"]].map(([v,l])=>`<option value="${v}" ${(p.training.krea_weights||"original")===v?"selected":""}>${l}</option>`).join("")}</select></label>`:""}<label class="field">Optimizer<select data-field="training.optimizer">${[["adamw8bit","AdamW8Bit"],["automagic3","Automagic3 (adaptive, experimental)"]].map(([v,l])=>`<option value="${v}" ${(p.training.optimizer||"adamw8bit")===v?"selected":""}>${l}</option>`).join("")}</select><small>For Automagic3, the learning rate is its starting value; it adapts during training.</small></label><label class="field">Sample shapes<select data-field="training.sample_layout">${['mixed','square','portrait','landscape'].map(v=>`<option value="${v}" ${(p.training.sample_layout||'mixed')===v?'selected':''}>${v}</option>`).join('')}</select></label></div></details><details><summary>Checks & export${c.warnings.length?' · '+c.warnings.length+' notes':''}</summary>${c.warnings.map(w=>`<p class="muted">${h(w)}</p>`).join('')}<p class="muted">Each run freezes its images, captions, configuration and preview prompts. Later edits won’t change that run.</p><p class="muted">${state.settings.integration==='standalone'||!state.settings.unload_comfy?'Automatic ComfyUI unloading is disabled. Ensure the training GPU has enough free memory.':'Studio checks for queued ComfyUI work before requesting a model unload. Keep new generations paused while training.'}</p><button data-action="export" ${state.current.preflight.errors.length?'disabled':''}>Export training bundle</button></details></details><div class="actions split"><button class="quiet" data-goto="dataset">← Dataset</button><button class="quiet" data-goto="runs">Runs & activity</button></div>`;
}
function runsView() {
  const jobs=[...state.current.jobs].sort((a,b)=>b.created-a.created);
  return `<section class="card"><div class="card-head"><div><h2>Runs & activity</h2><p>Caption jobs, immutable training bundles, logs and saved LoRAs.</p></div><span class="badge">${jobs.length} RUNS</span></div>${jobs.length?jobs.map(j=>{const active=['queued','running'].includes(j.status);return `<article class="run"><div class="run-head"><h3>${{preview_prompts:'Preview prompt generation',prepare:'Dataset preparation',verify:'Targeted visual checks',caption:'Vision captioning',detect:'Person count check',export:'Training bundle',train:'LoRA training'}[j.kind]}</h3><span class="badge ${j.status==='complete'?'good':j.status==='failed'?'bad':'warn'}">${h(j.status)}</span></div><small>${new Date(j.created*1000).toLocaleString()}${j.snapshot?' · '+h(j.snapshot.name):''}</small>${j.total?`<div class="progress"><progress max="${j.total}" value="${j.phase==='Reading images'?j.observed:j.completed}"></progress></div><p>${h(j.phase||'Processing')} · ${j.phase==='Reading images'?j.observed:j.completed} / ${j.total} images processed</p>`:''}${j.error?`<p class="danger">${h(j.error)}</p>`:''}${j.messages.map(m=>`<p>${h(m)}</p>`).join('')}<div class="actions">${j.kind==='train'?`<button data-results-job="${j.id}">View results</button><button class="primary" data-dashboard-job="${j.id}" data-dashboard-project="${j.project}">Open training dashboard</button>`:''}<button data-job="${j.id}">View details${j.bundle?' & files':''}</button>${active?`<button class="quiet danger" data-stop="${j.id}">Stop</button>`:''}</div></article>`}).join(''):'<div class="empty"><div class="empty-icon">↗</div><h3>Your activity will appear here</h3><p>Prepare captions or export a training bundle to start.</p></div>'}</section>`;
}
function modal(title,content) { $('#modal-content').innerHTML=`<div class="modal-head"><h2>${h(title)}</h2><button data-action="close-modal" aria-label="Close dialog">✕</button></div>${content}`; if(!$('#modal').open) $('#modal').showModal(); }
async function connections() {
  $('#system-status').open=false;
  await save(); const s=state.settings;
  modal('Settings',`<div class="actions split"><p class="muted">Check prerequisites and configure your workspace.</p><button type="button" id="run-setup">Run setup</button></div><form id="settings-form">${captionSettings(s)}<h3>LLM backend</h3><p class="muted">${s.integration==='standalone'?'Standalone · '+h(s.llm_provider||'OpenAI-compatible'):'Prompt Studio backend · recommended'}</p><button type="button" id="llm-settings">Backend settings</button><label class="checkbox"><input name="unload_comfy" type="checkbox" ${s.unload_comfy?'checked':''}>Request ComfyUI model unload before training when idle</label><div class="divider"></div><h3>AI Toolkit installation</h3><p class="muted">Use Run setup to check prerequisites or install a managed copy. Update existing installations independently.</p><div class="form-grid"><label class="field full">AI Toolkit folder<input name="toolkit_path" value="${h(s.toolkit_path)}" placeholder="Folder containing run.py"></label><label class="field full">AI Toolkit Python executable<input name="toolkit_python" value="${h(s.toolkit_python)}" placeholder="Full path to python.exe"></label></div><details><summary>Optional person detector</summary><label class="field">Python with Torchvision<input name="detector_python" value="${h(s.detector_python||'')}" placeholder="Leave blank to use AI Toolkit Python"><small>CPU only. Requires compatible torch, torchvision and Pillow. First check downloads ~14 MB of public weights. Counts are advisory; no identity matching.</small></label></details><div id="connection-result" class="note">${h(connectionText())}</div><div class="actions end extra-rules"><button type="submit" class="primary">Save & check connections</button></div></form>`);
  $('#settings-form').dataset.revision=s.settings_revision;
  $('#settings-form [name=caption_source]').addEventListener('change',()=>updateCaptionSettings($('#settings-form')));
}
function connectionText() {
  if(connectionError) return connectionError;
  if(connectionCheck) return 'Checking local connections…';
  if(!state.connection) return 'Connections have not been checked yet.';
  const c=state.connection, ps=c.promptstudio;
  if(state.settings.caption_source&&state.settings.caption_source!=='connected') return captionSourceLabel(state.settings.caption_source)+' · '+(c.captioner?.configured?'configured':'needs setup')+' · Text LLM '+(ps.available||c.standalone.available?'connected':'unavailable')+' · AI Toolkit '+(c.backend.available?'ready':'not configured');
  return (ps.available?`Prompt Studio · ${ps.provider} · ${ps.vision?.available?'vision ready':'vision unavailable: '+(ps.vision?.reason||'select a vision model')}`:c.standalone.available?'Standalone endpoint available'+(c.standalone.configured??state.settings.llm_model?'':' · choose a model ID'):'Vision connection unavailable')+' · AI Toolkit '+(c.backend.available?'ready':'not configured');
}
function renderConnectionStatus() {
  const c=state.connection, control=$('#system-status');
  let tone='checking', summary='Checking local connections…', vision='Checking…', toolkit='Checking…';
  if(connectionError) {
    tone='offline'; summary='Connection check failed'; vision=connectionError; toolkit='Status unavailable. Refresh to try again.';
  } else if(c&&!connectionCheck) {
    const ps=c.promptstudio, standalone=c.standalone;
    const dedicated=state.settings.caption_source&&state.settings.caption_source!=='connected';
    const visionAvailable=dedicated?c.captioner?.configured&&(ps.available||standalone.available&&(standalone.configured??Boolean(state.settings.llm_model))):ps.available?ps.vision?.available:standalone.available&&(standalone.configured??Boolean(state.settings.llm_model))&&standalone.vision?.available===true;
    tone=visionAvailable&&c.backend.available?'idle':!ps.available&&!standalone.available&&!c.backend.available?'offline':'attention';
    summary=tone==='idle'?'Connections available':tone==='offline'?'Connections unavailable':'Connections need attention';
    vision=dedicated?`${captionSourceLabel(state.settings.caption_source)} · ${c.captioner?.reason||'Configure captioning in Settings.'} Text LLM: ${ps.available?'Prompt Studio':standalone.available?'standalone endpoint':'unavailable'}.`:ps.available
      ? `Prompt Studio · ${ps.provider} · ${ps.model||'Loaded model'} — ${ps.vision?.available?'Vision ready':ps.vision?.reason||'Select a vision model in Prompt Studio.'}`
      : standalone.available
        ? `Standalone endpoint connected. ${standalone.vision?.available?'Vision capability reported.':standalone.configured?'Model configured. Use setup to test vision.':'Choose a vision model in Backend settings.'}`
        : (state.settings.integration==='promptstudio'?ps.reason:standalone.reason)||'Open Settings to configure a vision connection.';
    toolkit=c.backend.available?'Installation available. Model support is checked before training.':c.backend.reason||'Open Settings to configure AI Toolkit and its Python executable.';
  }
  control.dataset.state=tone;
  const label='System status: '+summary.toLowerCase();
  control.querySelector('summary').title=label;
  control.querySelector('summary').setAttribute('aria-label',label);
  $('#status-summary').textContent=summary;
  $('#vision-status').textContent=vision;
  $('#toolkit-status').textContent=toolkit;
  $('#status-checked').textContent=connectionCheck?'Checking connections…':connectionError?'Refresh to try again.':connectionChecked?'Last checked '+new Date(connectionChecked).toLocaleTimeString(): 'Connections have not been checked yet.';
  $('#refresh-connections').disabled=Boolean(connectionCheck);
  if($('#connection-result')) $('#connection-result').textContent=connectionText();
}
async function checkConnections() {
  if(connectionCheck) return connectionCheck;
  connectionError='';
  connectionCheck=api('/connection');
  renderConnectionStatus();
  try {state.connection=await connectionCheck;connectionChecked=Date.now();}
  catch(e) {connectionError=e.message;throw e;}
  finally {connectionCheck=null;renderConnectionStatus();}
}
function editImage(id) {
  const i=state.current.images.find(i=>i.id===id), status=imageState(i,state.current.intent_digest);
  modal('Review image & caption',`<div id="image-draft-banner"></div><form id="image-form" data-id="${i.id}" data-revision="${i.revision}"><div class="modal-grid"><div>${reviewPreparation(i,state.current.project)}${coverageControls(i,state.catalog,state.current.project)}<p class="muted">${h(i.name)}</p><span class="badge ${status.tone}">${status.label}</span><label class="checkbox"><input name="included" type="checkbox" ${i.included?'checked':''}>Include in training</label></div><div><label class="field">Training caption<textarea name="caption" maxlength="6000">${h(i.caption)}</textarea><small>Manual edits are preserved during automatic captioning.</small></label><label class="checkbox"><input name="reviewed" type="checkbox">Keep this caption · I reviewed it against the training image and learning rules</label>${reviewPanel(i,state.current.project,state.catalog)}${captionEvidence(i)}${i.warnings.length?`<div class="note warning">${i.warnings.map(w=>`<p>${h(w)}</p>`).join('')}</div>`:''}<details><summary>Original visual observations · ${h(captionSourceLabel(i.caption_source))}</summary><p>${h(i.observations||'No vision analysis yet.')}</p></details><details><summary>Caption history · ${i.history.length}</summary>${[...i.history].reverse().map(v=>`<p><small>${new Date(v.at*1000).toLocaleString()} · ${h(v.source)}</small><br>${h(v.caption||'(Empty)')}</p>`).join('')}</details></div></div><div class="actions end extra-rules"><button type="submit" name="next" value="1">Save & next</button><button type="submit" class="primary">Save image review</button></div></form>`);
  if(localStorage.getItem(imageDraftKey(id))) $('#image-draft-banner').innerHTML='<div class="note warning">An unsaved image review is available. Restoring uses the latest revision shown here; review it before saving.<div class="actions"><button data-action="restore-image-draft">Restore draft</button><button data-action="discard-image-draft">Discard draft</button></div></div>';
  if(!preparationLocked(state.current.project))bindObjectEditor($('#image-form'),i);
  renderPreparationLock();
  updatePreparationPreview();
}
let previewSequence=0, previewTimer=null;
function imageDraftKey(id) {return 'lora-studio.image-draft.'+id;}
function imageFormValue(form) {
  return {...(readObjectForm(form)?{object_target:readObjectForm(form)}:{}),revision:Number(form.dataset.revision),caption:form.elements.caption.value,included:form.elements.included.checked,reviewed:form.elements.reviewed.checked,
    preparation:{mode:form.elements.prep_mode.value,x:Number(form.elements.prep_x.value),y:Number(form.elements.prep_y.value)},
    coverage_kind:state.current.project.kind, coverage_override:Object.fromEntries(Object.keys(state.catalog.types.find(t=>t.id===state.current.project.kind).coverage).map(k=>[k,form.elements['coverage_'+k]?.value||'']).filter(([,v])=>v)),
    ...(form.elements.accept_verification?{accept_verification:form.elements.accept_verification.checked}:{}),
    ...(form.elements.person_count?{person_count:form.elements.person_count.value===''?null:Number(form.elements.person_count.value)}:{}),accept_advisories:form.elements.accept_advisories.checked};
}
async function updatePreparationPreview() {
  const form=$('#image-form');if(!form)return;
  const sequence=++previewSequence;
  try {
    const result=await api('/images/'+form.dataset.id+'/preview','POST',{preparation:imageFormValue(form).preparation,object_target:readObjectForm(form)});
    if(sequence!==previewSequence||form!==$('#image-form'))return;
    const p=result.plan;
    if($('#object-mask-figure')){$('#object-mask-figure').hidden=!result.mask_png;if(result.mask_png)$('#object-mask-preview').src='data:image/png;base64,'+result.mask_png;}
    $('#prepared-preview').src='data:image/png;base64,'+result.png;
    $('#preparation-result').textContent=`${p.source_width} × ${p.source_height} → ${p.width} × ${p.height} · ${p.source_region?'target crop + ':''}${p.mode==='crop'?p.removed_percent+'% fit adjustment':'padding around selected region'}`;
    const original=new Image();original.onload=()=>{if(sequence===previewSequence&&$('#crop-overlay')){drawCrop($('#crop-overlay'),original,p,form.elements.show_boxes?.checked?state.current.images.find(i=>i.id===form.dataset.id)?.detection:null);drawObjectMask($('#crop-overlay'),original,form);}};original.src='/api/images/'+form.dataset.id;
    form.elements.prep_x.disabled=form.elements.prep_y.disabled=preparationLocked(state.current.project)||p.mode!=='crop';
  }catch(e){if(form===$('#image-form'))$('#preparation-result').textContent=e.message;}
}
async function jobDetails(id) {
  const {job:j,files}=await api('/jobs/'+id);
  modal('Run details',`<p><span class="badge">${h(j.status)}</span> ${h(j.kind)} · ${new Date(j.created*1000).toLocaleString()}</p>${j.error?`<div class="note error">${h(j.error)}</div>`:''}${j.messages.map(m=>`<p>${h(m)}</p>`).join('')}<div class="file-list">${files.map(f=>`<a href="/api/jobs/${id}/file?path=${encodeURIComponent(f.path)}" target="_blank" rel="noopener">${h(f.path)} <small>(${Math.ceil(f.size/1024)} KB)</small></a>`).join('')||'<p class="muted">No exported files yet.</p>'}</div>${files.some(f=>f.path==='training.log')?'<pre id="log-preview">Loading latest log…</pre>':''}<div class="actions end extra-rules">${j.status==='interrupted'&&j.kind==='train'?`<button data-recover="${id}">Check previous trainer process</button>`:''}<button data-job="${id}">Refresh details</button></div>`);
  if($('#log-preview')) {const log=await fetch(`/api/jobs/${id}/file?path=training.log`).then(r=>r.text());if($('#log-preview'))$('#log-preview').textContent=log.slice(-18000)||'Waiting for trainer output…';}
}
async function importImages(files) {
  if(preparationLocked(state.current.project))throw new Error('Preparation is locked. Use Copy to new run to import or edit images.');
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
  if(name==='copy-run'){
    await save();const project=await api(`/projects/${state.current.project.id}/copy`,'POST',{});
    state.projects.unshift(project);state.tab='setup';dashboardJob=null;resultsJob=null;
    await selectProject(project.id);toast('Copied to a fresh project. All preparation is editable.');return;
  }
  if(name==='verify-image') {
    const form=$('#image-form'),item=state.current.images.find(i=>i.id===form.dataset.id),value=imageFormValue(form);
    const savedPrep=item.preparation||{};
    if(value.caption!==item.caption||value.preparation.mode!==(savedPrep.mode||'auto')||value.preparation.x!==(savedPrep.x??0.5)||value.preparation.y!==(savedPrep.y??0.5)) throw new Error('Save caption and crop edits before checking this image. Your draft is retained.');
    const checks=state.catalog.types.find(t=>t.id===state.current.project.kind).checks.filter(k=>form.elements['verify_'+k]?.checked);
    await api('/projects/'+state.current.project.id+'/verify','POST',{image_ids:[item.id],checks,focus:form.elements.verify_focus.value});
    $('#modal').close();state.tab='runs';await refresh();render();toast('Visual check started; the caption stays unchanged.');return;
  }

  if(name==='apply-preparation') {await save();await refresh();render();toast('Preparation refreshed. Review any captions affected by changed crops.');return;}
  if(name==='recommended-resolution') {const p=state.current.project;p.training.resolution=state.catalog.models.find(m=>m.id===p.model).preparation.default;changed();await save();await refresh();render();return;}
  if(name==='restore-image-draft'||name==='discard-image-draft') {
    const f=$('#image-form'),key=imageDraftKey(f.dataset.id);
    if(name==='restore-image-draft') {const d=JSON.parse(localStorage.getItem(key));restoreObjectForm(f,d.object_target);f.elements.caption.value=d.caption;f.elements.included.checked=d.included;f.elements.reviewed.checked=false;f.elements.prep_mode.value=d.preparation.mode;f.elements.prep_x.value=d.preparation.x;f.elements.prep_y.value=d.preparation.y;if(f.elements.person_count)f.elements.person_count.value=d.person_count??'';f.elements.accept_advisories.checked=false;for(const [k,v] of Object.entries((d.coverage_kind||'character')===state.current.project.kind?d.coverage_override||{}:{})){if(f.elements['coverage_'+k])f.elements['coverage_'+k].value=v;}updatePreparationPreview();}
    else localStorage.removeItem(key);
    $('#image-draft-banner').innerHTML='';return;
  }
  if(name==='new') {await save();const p=await api('/projects','POST',{});state.projects.unshift(p);state.tab='setup';await selectProject(p.id);return;}
  if(name==='close-modal') {$('#modal').close();return;}
  if(name==='pick-images'||name==='pick-folder') {$(name==='pick-images'?'#image-input':'#folder-input').click();return;}
  if(name==='restore') {const revision=state.current.project.revision;state.current.project={...state.recovered,revision};state.recovered=false;changed();render();return;}
  if(name==='discard') {localStorage.removeItem(draftKey(state.current.project.id));state.recovered=false;render();return;}
  if(name==='download-draft') {const blob=new Blob([JSON.stringify(state.recovered,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='lora-project-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);return;}
  if(name==='apply-preset'){await save();await api(`/projects/${state.current.project.id}/preset`,'POST',{preset:$('#training-preset').value,revision:state.current.project.revision});await refresh();render();toast('Baseline applied. You can adjust any setting.');return;}
  if(name==='generate-previews'){await save();await api(`/projects/${state.current.project.id}/sample-prompts`,'POST',{});await refresh();render();toast('Generating target-aware preview prompts.');return;}
  if(name==='restore-preview-prompts'){const p=state.current.project,row=p.sample_prompt_history?.[Number($('#prompt-history').value)];if(row){p.sample_prompts=row.text;changed();render();}return;}
  if(name==='edit-previews'){await save();await loadPreviewSuite();const p=state.current.project;p.sample_prompts=(state.previewSuite?.prompts||[]).map(p=>p.text).join('\n');p.sample_prompt_mode='manual';changed();render();return;}
  if(['prepare','caption','reprocess-captions','detect','objects','export','train'].includes(name)) {
    await save();if(state.recovered)throw new Error('Restore or discard the unsaved draft before starting a job.');
    await api(`/projects/${state.current.project.id}/${name==='reprocess-captions'?'caption':name}`,'POST',name==='reprocess-captions'?{reuse_observations:true}:{});state.tab=name==='train'?'dashboard':'runs';dashboardJob=null;await refresh();render();toast(name==='train'?'Training launch queued. Follow progress in Dashboard.':'Job started.');
  }
}
document.addEventListener('click',async e=>{
  if(!e.target.closest('#system-status')) $('#system-status').open=false;
  const b=e.target.closest('button');if(!b||b.disabled)return;
  try {
    if(await comparisons.click(b,resultsJob,resultsData))return;
    if(b.dataset.tab||b.dataset.goto)comparisons.leave();
    if(b.id==='llm-settings'){await save();return await openSetup(2);}
    if(b.id==='run-setup'){await save();return await openSetup();}
    if(b.id==='new-project') return await action('new');
    if(b.id==='connections'||b.dataset.action==='settings') return await connections();
    if(b.id==='refresh-connections') return await checkConnections();
    if(b.dataset.deleteProject) return await confirmProjectDeletion(b.dataset.deleteProject);
    if(b.dataset.confirmDeleteProject) return await deleteProject(b);
    if(b.dataset.project) return await selectProject(b.dataset.project);
    if(b.dataset.dashboardJob){await save();dashboardJob=b.dataset.dashboardJob;state.tab='dashboard';await selectProject(b.dataset.dashboardProject);return;}
    if(b.dataset.extendJob){
      const run=await api(`/jobs/${b.dataset.extendJob}/monitor`),interval=run.config.config?.process?.[0]?.save?.save_every;
      modal('Continue training',run.capabilities.extend?`<p>Continue from step ${run.total.toLocaleString()} using this run’s frozen inputs and saved optimizer. Each additional checkpoint adds ${interval.toLocaleString()} steps. Previous outputs are retained.</p><label class="field">Additional checkpoints<input id="continue-checkpoints" type="number" min="1" max="1000" step="1" value="1" required></label><div class="actions end"><button data-action="close-modal">Cancel</button><button class="primary" data-confirm-extend="${h(run.job.id)}">Continue training</button></div>`:`<p>${h(run.resume_error||'A complete final checkpoint and optimizer save are required.')}</p>`);return;
    }
    if(b.dataset.confirmExtend){
      const input=$('#continue-checkpoints');if(!input.reportValidity())return;
      b.disabled=true;try{await api(`/jobs/${b.dataset.confirmExtend}/extend`,'POST',{checkpoints:Number(input.value)});$('#modal').close();dashboardJob=b.dataset.confirmExtend;state.tab='dashboard';await refresh();render();toast('Continuing training from the final checkpoint.');}finally{b.disabled=false;}return;
    }
    if(b.dataset.resultsJob){await save();resultsJob=b.dataset.resultsJob;resultsData=null;resultsError='';state.tab='results';render();return;}
    if(b.hasAttribute('data-delete-checkpoint')){
      b.disabled=true;try{
        const job=resultsJob,path=b.dataset.deleteCheckpoint;
        const plan=await api(`/jobs/${job}/checkpoint-delete-plan`,'POST',{path});
        modal('Delete checkpoint?',`<p>Permanently delete <strong>${h(plan.checkpoint)}</strong> and its related files?</p><p>This removes ${plan.files.toLocaleString()} local files, including its previews, thumbnails and sidecars when present.${plan.comparisons?` It also removes ${plan.comparisons} entire saved comparison${plan.comparisons===1?'':'s'} containing this checkpoint, including all their images and retained weights.`:''}</p>${plan.optimizer?'<p class="note warning">This includes the latest optimizer state in this output folder. It will no longer be possible to resume from this save.</p>':''}${plan.shared_previews?'<p>Previews shared with another retained checkpoint stay available.</p>':''}<p>The dataset, captions, run history and other training checkpoints stay intact. This cannot be undone.</p><div class="actions end"><button data-action="close-modal">Cancel</button><button class="danger" data-confirm-delete-checkpoint="${h(path)}" data-delete-job="${h(job)}" data-delete-token="${h(plan.token)}">Delete checkpoint</button></div>`);
      }finally{b.disabled=false;}return;
    }
    if(b.hasAttribute('data-confirm-delete-checkpoint')){
      b.disabled=true;try{
        await api(`/jobs/${b.dataset.deleteJob}/delete-checkpoint`,'POST',{path:b.dataset.confirmDeleteCheckpoint,token:b.dataset.deleteToken});
        $('#modal').close();resultsData=null;resultsError='';await loadResults(resultsJob);render();toast('Checkpoint and related files deleted.');
      }finally{b.disabled=false;}return;
    }
    if(b.hasAttribute('data-result-refresh')){resultsData=null;resultsError='';await refresh();render();return;}
    if(b.hasAttribute('data-result-location')){b.disabled=true;try{await api(`/jobs/${resultsJob}/open-location`,'POST',{path:b.dataset.resultLocation||null});toast('Output folder opened.');}finally{b.disabled=false;}return;}
    if(b.dataset.tab||b.dataset.goto) {await save();state.tab=b.dataset.tab||b.dataset.goto;if(state.tab==='results'){resultsData=null;resultsError='';}await refresh();render();return;}
    if(b.dataset.kind) {state.current.project.kind=b.dataset.kind;changed();render();return;}
    if(b.dataset.action){if(b.dataset.action==='copy-run')b.disabled=true;try{return await action(b.dataset.action);}finally{if(b.dataset.action==='copy-run')b.disabled=false;}}
    if(b.dataset.image) {await save();await refresh();return editImage(b.dataset.image);}
    if(b.dataset.job) return await jobDetails(b.dataset.job);
    if(b.dataset.recover) {await api(`/jobs/${b.dataset.recover}/recover`,'POST',{});await refresh();render();return await jobDetails(b.dataset.recover);}
    if(b.dataset.stop) {const j=state.current.jobs.find(j=>j.id===b.dataset.stop);modal('Stop this job?',`<p>${j.kind==='detect'?'Stops the CPU detector. Completed results are preserved.':['prepare','caption','verify'].includes(j.kind)?'Stops after the current vision request returns. Completed captions stay saved.':'The trainer will stop. Work since the last checkpoint may be lost.'}</p><div class="actions end"><button data-action="close-modal">Keep running</button><button class="danger" data-confirm-stop="${j.id}">Stop job</button></div>`);return;}
    if(b.dataset.confirmStop) {await api(`/jobs/${b.dataset.confirmStop}/stop`,'POST',{});$('#modal').close();toast('Stop requested.');}
  }catch(err){toast(err.message,true);}
});
document.addEventListener('input',e=>{
  if(preparationLocked(state.current?.project)&&e.target.closest('#content,#image-form')){if(e.target.name==='show_boxes')updatePreparationPreview();return;}
  const form=e.target.closest('#image-form');
  if(form) {if(['caption','prep_mode','prep_x','prep_y'].includes(e.target.name)&&form.elements.accept_verification)form.elements.accept_verification.checked=false;if(e.target.name==='show_boxes')updatePreparationPreview();if(e.target.name.startsWith('prep_')||e.target.name.startsWith('object_')){form.elements.reviewed.checked=false;form.elements.accept_advisories.checked=false;clearTimeout(previewTimer);previewTimer=setTimeout(updatePreparationPreview,180);}try{localStorage.setItem(imageDraftKey(form.dataset.id),JSON.stringify(imageFormValue(form)));}catch{toast('Image draft storage unavailable. Keep this review open until saved.',true);}return;}
  if(e.target.dataset.preparation){state.current.project.preparation={...(state.current.project.preparation||{}),[e.target.dataset.preparation]:e.target.value};changed();return;}
  if(e.target.dataset.loraMetadata){setMetadataField(state.current.project,e.target.dataset.loraMetadata,e.target.value);changed();return;}
  const key=e.target.dataset.field;if(!key)return;
  if(key.startsWith('training.'))state.current.project.training[key.split('.')[1]]=['training.sample_layout','training.optimizer','training.krea_weights'].includes(key)?e.target.value:Number(e.target.value);else state.current.project[key]=e.target.value;
  changed();if(key==='sample_prompt_mode'){save().then(async()=>{await loadPreviewSuite();render();}).catch(error=>toast(error.message,true));}else if(key==='model'||key==='training_mode'||key==='style_family'||key==='concept_family')render();
});
document.addEventListener('change',async e=>{
  if(e.target.id==='results-run'){comparisons.leave();resultsJob=e.target.value;resultsData=null;resultsError='';render();return;}
  if(e.target.id==='dataset-filter'){datasetFilter=e.target.value;render();return;}
  if(e.target.dataset.attribute){state.current.project.choices[e.target.dataset.attribute]=e.target.value;changed();}
  if(e.target.id==='image-input'||e.target.id==='folder-input'){try{await importImages([...e.target.files]);}catch(err){toast(err.message,true);}}
});
document.addEventListener('submit',async e=>{
  e.preventDefault();const form=e.target,button=form.querySelector('[type=submit]');button.disabled=true;
  try{
    if(form.id==='settings-form'){const data=Object.fromEntries(new FormData(form));data.settings_revision=Number(form.dataset.revision);data.unload_comfy=form.elements.unload_comfy.checked;data.caption_gpu=Number(form.elements.caption_gpu.value);if(connectionCheck)await connectionCheck.catch(()=>{});state.settings=await api('/settings','PUT',data);form.dataset.revision=state.settings.settings_revision;await checkConnections();await refresh();render();toast('Connections saved.');}
    if(form.id==='image-form'){const sequence=filterImages(state.current.images,datasetFilter),position=sequence.findIndex(i=>i.id===form.dataset.id),next=sequence[(position+1)%sequence.length]?.id;const value=imageFormValue(form);localStorage.setItem(imageDraftKey(form.dataset.id),JSON.stringify(value));await api('/images/'+form.dataset.id,'PUT',value);localStorage.removeItem(imageDraftKey(form.dataset.id));$('#modal').close();await refresh();render();toast('Image review saved.');if(e.submitter?.name==='next'&&next&&next!==form.dataset.id)editImage(next);}
  }catch(err){toast(err.message,true);}finally{button.disabled=false;}
});
document.addEventListener('dragover',e=>{const zone=e.target.closest('#drop-zone');if(zone){e.preventDefault();zone.classList.add('drag');}});
document.addEventListener('dragleave',e=>{e.target.closest('#drop-zone')?.classList.remove('drag');});
document.addEventListener('drop',async e=>{if(e.target.closest('#drop-zone')){e.preventDefault();try{await importImages([...e.dataTransfer.files]);}catch(err){toast(err.message,true);}}});
window.addEventListener('beforeunload',e=>{if(state.dirty||state.saving){e.preventDefault();e.returnValue='';}});
$('#system-status').addEventListener('toggle',()=>{if($('#system-status').open&&state.settings&&Date.now()-connectionChecked>30000)checkConnections().catch(()=>{});});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$('#system-status').open){$('#system-status').open=false;$('#system-status summary').focus();}});
async function boot(){const b=await api('/bootstrap');Object.assign(state,b);renderProjects();render();const active=navigation.get('project')||localStorage.getItem('lora-studio.active');state.tab=['setup','dataset','training','dashboard','runs','results'].includes(navigation.get('tab'))?navigation.get('tab'):'setup';dashboardJob=navigation.get('job');resultsJob=navigation.get('job');if(state.projects.length)await selectProject(state.projects.some(p=>p.id===active)?active:state.projects[0].id);refreshActivity();checkConnections().catch(()=>{});if(b.onboarding?.disposition==='new'||['queued','running','interrupted'].includes(b.onboarding?.install?.status))await openSetup();}
boot().catch(e=>{connectionError=e.message;renderConnectionStatus();toast(e.message,true);saveLabel('Could not load workspace');});
setInterval(async()=>{
  if(!state.current||state.saving)return;
  const previous=JSON.stringify(state.current.project.preparation_lock), active=state.current.jobs.some(j=>['queued','running'].includes(j.status));
  try{await refresh();if(state.dirty||$('#modal').open)return;
    if(previous!==JSON.stringify(state.current.project.preparation_lock)){render();return;}
    if(!active)return;
    if(state.tab==='results'){if(resultsData?.job.id===resultsJob)await loadResults(resultsJob);else render();}
    else if(['runs','dataset','training'].includes(state.tab))render();
  }catch{}
},2500);

setInterval(()=>{if(!document.hidden)refreshActivity();},5000);

function renderResults(){
  if(comparisons.visible()){comparisons.render();return;}
  const runs=state.current.jobs.filter(j=>j.kind==='train').sort((a,b)=>b.created-a.created);
  const selected=runs.find(j=>j.id===resultsJob)||runs.find(j=>j.status==='complete')||runs[0];
  resultsJob=selected?.id||null;updateNavigation();
  $('#content').innerHTML=resultsView(state.current.jobs,resultsJob,resultsData,resultsError);
  if(selected&&!resultsError&&resultsData?.job.id!==resultsJob)loadResults(resultsJob);
}
async function loadResults(id){
  const request=++resultsRequest,project=state.current.project.id;let updated=false;
  try{const data=await api(`/jobs/${id}/results`);if(request!==resultsRequest||state.current?.project.id!==project||resultsJob!==id)return;updated=Boolean(resultsError)||JSON.stringify(resultsData)!==JSON.stringify(data);resultsData=data;resultsError='';}
  catch(error){if(request!==resultsRequest||state.current?.project.id!==project||resultsJob!==id)return;updated=resultsError!==error.message;resultsError=error.message;}
  if(state.tab==='results'&&updated)render();
}
