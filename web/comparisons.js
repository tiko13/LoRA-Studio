import {escapeHTML as h} from './domain.js';

const imageURL=(id,name)=>`/api/comparisons/${encodeURIComponent(id)}/images/${encodeURIComponent(name)}`;
const cellSizeKey='lora-studio.comparison-cell-size';
const normalizeCellSize=value=>Number.isFinite(Number(value))&&Number(value)>0?Math.max(120,Math.min(420,Math.round(Number(value)/20)*20)):220;
export function comparisonList(comparisons=[]){
  return `<section class="card comparison-history"><h3>Checkpoint comparisons</h3>${comparisons.length?comparisons.map(c=>`<button class="comparison-saved" data-comparison-open="${h(c.id)}"><span><strong>${h(c.workflow_name)}</strong><small>${h(new Date(c.created*1000).toLocaleString())} · ${c.columns.length} columns × ${c.settings.seeds.length} seeds</small></span><span>${c.cells.filter(x=>x.status==='complete').length}/${c.cells.length} · ${h(c.status)}</span></button>`).join(''):'<p class="muted">Compare saved checkpoints with the same prompt and seeds. Generated comparisons stay with this run.</p>'}</section>`;
}

export function comparisonGrid(c,cellSize=220){
  cellSize=normalizeCellSize(cellSize);
  const active=['preparing','running','stopping'].includes(c.status),done=c.cells.filter(x=>x.status==='complete').length;
  return `<section class="card comparison-view" style="--comparison-cell-size:${cellSize}px"><div class="card-head"><div><span class="eyebrow">CHECKPOINT COMPARISON</span><h2>${h(c.workflow_name)}</h2><p>${done}/${c.cells.length} images · ${h(c.status)} · strength ${h(c.settings.strength)}</p></div><button data-comparison-back>Back to results</button></div>
    ${c.error?`<p class="note warning" role="alert">${h(c.error)}</p>`:''}
    <div class="actions">${active?`<button data-comparison-action="stop" ${c.status==='stopping'?'disabled':''}>Stop after current image</button>`:`${done<c.cells.length?'<button class="primary" data-comparison-action="retry">Resume / retry failed</button>':''}<button data-comparison-rerun>Re-run comparison</button>`}<a class="button" href="/api/comparisons/${h(c.id)}/grid" download>Download grid</a></div>
    <details><summary>Prompt & settings</summary><p class="comparison-prompt">${h(c.settings.prompt)}</p><p class="muted">${h(Object.values(c.settings.models).join(' · ')||'Workflow model')} · ${c.settings.width?`${c.settings.width} × ${c.settings.height}`:'Workflow resolution'} · ${c.settings.steps??'Workflow'} steps · CFG ${c.settings.cfg??'workflow'}</p></details>
    <p class="muted">The same seeds and prompt are used for every checkpoint. Click an image to enlarge it.</p>
    <div class="comparison-toolbar"><label for="comparison-cell-size">Cell size</label><input id="comparison-cell-size" type="range" min="120" max="420" step="20" value="${cellSize}" aria-valuetext="${cellSize} pixels"><output for="comparison-cell-size">${cellSize} px</output></div>
    <div class="comparison-scroll" tabindex="0" aria-label="Checkpoint by seed comparison"><table class="comparison-grid"><thead><tr><th scope="col">Seed</th>${c.columns.map(col=>`<th scope="col">${h(col.label)}</th>`).join('')}</tr></thead><tbody>${c.settings.seeds.map(seed=>`<tr><th scope="row">${seed}</th>${c.columns.map((_,col)=>{const cell=c.cells.find(x=>x.column===col&&x.seed===seed);return `<td data-cell="${h(cell.id)}">${cell.images.length?cell.images.map(name=>`<button class="comparison-image" data-image-preview="${imageURL(c.id,name)}" aria-label="Enlarge ${h(c.columns[col].label)}, seed ${seed}"><img src="${imageURL(c.id,name)}" alt="${h(c.columns[col].label)}, seed ${seed}" loading="lazy"></button>`).join(''):`<div class="comparison-placeholder"><span>${h(cell.status)}</span>${cell.error?`<small>${h(cell.error)}</small>`:''}</div>`}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div></section>`;
}

export function checkpointComparison({api,modal,toast,refreshResults}){
  let selected=null,activeJob=null,polling=false,generation=0,dialogRequest=0;
  let cellSize=220;try{cellSize=normalizeCellSize(localStorage.getItem(cellSizeKey));}catch{}
  const drafts=new Map();
  const $=s=>document.querySelector(s);
  function draw(data){
    const previous=$('.comparison-scroll'),left=previous?.scrollLeft,top=previous?.scrollTop;
    const previousStatus=selected;selected=data;
    if(previousStatus&&previousStatus.id===data.id&&JSON.stringify(previousStatus)===JSON.stringify(data))return;
    const next=document.createElement('div');next.innerHTML=comparisonGrid(data,cellSize);
    if(previous?.dataset.comparison===data.id){
      for(const cell of next.querySelectorAll('[data-cell]')){const old=previous.querySelector(`[data-cell="${cell.dataset.cell}"]`);if(old&&old.innerHTML!==cell.innerHTML)old.innerHTML=cell.innerHTML;}
      // Keep the slider, focus, open settings and grid mounted during progress updates.
      const view=$('.comparison-view');
      for(const selector of ['.card-head','.actions'])view.querySelector(selector).replaceWith(next.querySelector(selector));
      const oldError=view.querySelector('[role=alert]'),newError=next.querySelector('[role=alert]');
      if(oldError)oldError.remove();if(newError)view.querySelector('.actions').before(newError);
      return;
    }
    $('#content').replaceChildren(...next.childNodes);
    $('.workspace').classList.add('comparison-workspace');
    $('.comparison-scroll').dataset.comparison=data.id;
    if(previous){$('.comparison-scroll').scrollLeft=left;$('.comparison-scroll').scrollTop=top;}
  }
  document.addEventListener('input',event=>{
    if(event.target.id!=='comparison-cell-size'||!selected)return;
    cellSize=normalizeCellSize(event.target.value);
    $('.comparison-view').style.setProperty('--comparison-cell-size',`${cellSize}px`);
    event.target.setAttribute('aria-valuetext',`${cellSize} pixels`);
    $('.comparison-toolbar output').textContent=`${cellSize} px`;
    try{localStorage.setItem(cellSizeKey,String(cellSize));}catch{}
  });
  async function open(identity){
    const version=++generation;selected=null;
    const data=await api('/comparisons/'+identity);if(version!==generation)return;
    activeJob=data.job;draw(data);
  }
  function leave(){generation++;selected=null;activeJob=null;$('.workspace')?.classList.remove('comparison-workspace');}
  async function poll(){
    if(!selected||polling||document.hidden)return;
    const id=selected.id,version=generation;polling=true;
    try{const data=await api('/comparisons/'+id);if(version===generation&&selected?.id===id)draw(data);}
    catch(e){if(version===generation)toast(e.message,true);}finally{polling=false;}
  }
  setInterval(poll,2500);
  async function setup(job,results,initial=null){
    activeJob=job;
    const request=++dialogRequest;
    modal('Compare checkpoints','<p id="comparison-loading" role="status">Loading compatible Create workflows from Prompt Studio…</p>');
    const loading=$('#comparison-loading');
    let options;
    try{options=await api(`/jobs/${job}/comparison-options`);}catch(e){const stale=e.status===404&&/route not found/i.test(e.message);if(request===dialogRequest&&loading.isConnected&&$('#modal').open)modal('Compare checkpoints',`<p class="note warning">${h(stale?'LoRA Studio is still running an older backend. Stop and relaunch LoRA Studio, then refresh this page.':e.message)}</p><p>${stale?'Reopening the launcher while Studio is running only opens its existing window.':'Check the Prompt Studio connection in Settings. Saved comparison images remain available offline.'}</p>`);return;}
    if(!$('#modal').open||request!==dialogRequest||!loading.isConnected)return;
    if(!options.workflows.length){modal('Compare checkpoints','<p>No compatible Create workflows are available. Open Prompt Studio and refresh its workflows. A workflow needs a matching LoRA type, a controllable seed, and an image result. Edit and reference-image workflows are excluded.</p>');return;}
    const ready=results.checkpoints.filter(c=>c.ready).sort((a,b)=>(a.step??Infinity)-(b.step??Infinity)||Number(a.final)-Number(b.final));
    let stored={};try{stored=JSON.parse(localStorage.getItem('lora-studio.comparison-draft.'+job)||'{}')||{};}catch{}
    const draft=initial||drafts.get(job)||stored;
    const chosen=draft.checkpoints??ready.map(c=>c.path);
    modal('Compare checkpoints',`<form id="comparison-form"><p class="muted">Requires Prompt Studio · Create workflows only</p>
      <fieldset class="comparison-checkpoints"><legend>Checkpoints</legend><div class="actions"><button type="button" data-select-checkpoints="all">Select all</button><button type="button" data-select-checkpoints="none">Select none</button></div><div class="checkpoint-options">${ready.map(c=>`<label><input type="checkbox" name="checkpoint" value="${h(c.path)}" ${chosen.includes(c.path)?'checked':''}><span>${c.final?'Final · ':''}${c.step===null?h(c.name):`Step ${c.step.toLocaleString()}`}<small>${h(c.name)}</small></span></label>`).join('')}</div></fieldset>
      <label class="field">Create workflow<select name="workflow">${options.workflows.map(p=>`<option value="${h(p.id)}" ${draft.workflow===p.id?'selected':''}>${h(p.name)}</option>`).join('')}</select></label><div id="comparison-workflow-controls"></div>
      <label class="field">LoRA strength<input name="strength" type="number" min="-100" max="100" step="0.05" value="${h(draft.strength??1)}" required></label>
      <label class="field">Prompt<textarea name="prompt" rows="4" maxlength="16000" required>${h(draft.prompt??options.prompt)}</textarea></label>${options.trigger?`<div class="actions"><span class="muted">Trigger: ${h(options.trigger)}</span><button type="button" data-insert-trigger>Insert trigger</button></div>`:''}
      <label class="field">Seeds<input name="seeds" value="${h((draft.seeds||[42,12345]).join(', '))}" required><small>Comma-separated whole numbers. Each checkpoint uses the same seeds.</small></label><div class="actions"><button type="button" data-random-seeds>Randomize seeds</button></div>
      <label class="comparison-check"><input type="checkbox" name="baseline" ${draft.baseline?'checked':''}> Include baseline without the trained LoRA</label>
      <details class="comparison-advanced"><summary>Generation settings</summary><p class="muted">Leave blank to use the workflow defaults. Supporting LoRAs remain fixed.</p><div class="form-grid"><label class="field">Width<input name="width" type="number" min="64" max="4096" step="32" placeholder="Workflow default" value="${h(draft.width??'')}"></label><label class="field">Height<input name="height" type="number" min="64" max="4096" step="32" placeholder="Workflow default" value="${h(draft.height??'')}"></label><label class="field">Steps<input name="steps" type="number" min="1" max="200" placeholder="Workflow default" value="${h(draft.steps??'')}"></label><label class="field">CFG<input name="cfg" type="number" min="0" max="100" step="0.1" placeholder="Workflow default" value="${h(draft.cfg??'')}"></label></div></details>
      <p id="comparison-count" aria-live="polite"></p><p id="comparison-error" class="warning" role="alert"></p><div class="actions end"><button type="submit" class="primary">Generate comparison</button></div></form>`);
    const form=$('#comparison-form');let busy=false;
    const modelDraft={...(draft.models||{})};
    function profile(){return options.workflows.find(p=>p.id===form.elements.workflow.value);}
    function workflow(){
      const p=profile();
      $('#comparison-workflow-controls').innerHTML=`${p.comparison_loras.length>1?`<label class="field">LoRA target<select name="loader">${p.comparison_loras.map(n=>`<option value="${h(n.id)}" ${draft.loader===String(n.id)?'selected':''}>${h(n.loraType)} · node ${h(n.id)}</option>`).join('')}</select></label>`:`<input name="loader" type="hidden" value="${h(p.comparison_loras[0].id)}">`}${p.modelNodes?.length?p.modelNodes.map(n=>`<label class="field">Model${p.modelNodes.length>1?` · ${h(n.modelType)}`:''}<select data-comparison-model="${h(n.id)}">${n.choices.map(m=>`<option value="${h(m.name)}" ${(modelDraft[n.id]??n.modelName)===m.name?'selected':''}>${h(m.label||m.name)}</option>`).join('')}</select></label>`).join(''):'<p class="muted">Model: fixed by the workflow</p>'}`;
      const adjustable=p.comparison_samplers.some(id=>p.snapshot.output[id].class_type==='KCPP_PromptStudioSampler');
      for(const name of ['steps','cfg']){form.elements[name].disabled=!adjustable;form.elements[name].placeholder=adjustable?'Workflow default':'Fixed by workflow';}
      for(const name of ['width','height']){form.elements[name].disabled=!p.comparison_resolution;form.elements[name].placeholder=p.comparison_resolution?'Workflow default':'Fixed by workflow';}
    }
    function read(){const value={checkpoints:[...form.querySelectorAll('[name=checkpoint]:checked')].map(e=>e.value),workflow:form.elements.workflow.value,
      loader:form.elements.loader.value,models:Object.fromEntries([...form.querySelectorAll('[data-comparison-model]')].map(e=>[e.dataset.comparisonModel,e.value])),
      strength:Number(form.elements.strength.value),prompt:form.elements.prompt.value,seeds:form.elements.seeds.value.split(/[\s,]+/).filter(Boolean).map(Number),baseline:form.elements.baseline.checked};
      for(const name of ['width','height','steps','cfg'])if(!form.elements[name].disabled&&form.elements[name].value!=='')value[name]=Number(form.elements[name].value);
      return value;
    }
    function update(){const v=read(),columns=v.checkpoints.length+Number(v.baseline),seeds=new Set(v.seeds).size,count=columns*seeds;drafts.set(job,v);try{localStorage.setItem('lora-studio.comparison-draft.'+job,JSON.stringify(v));}catch{}
      $('#comparison-count').textContent=`${v.checkpoints.length} checkpoints${v.baseline?' + baseline':''} × ${seeds} seeds = ${count} images`;
      form.querySelector('[type=submit]').disabled=busy||!v.checkpoints.length||!seeds||count>options.max_cells;
      if(count>options.max_cells)$('#comparison-count').textContent+=` · Maximum ${options.max_cells}`;
    }
    workflow();update();
    form.oninput=update;
    form.onchange=e=>{if(e.target.name==='workflow'){for(const el of form.querySelectorAll('[data-comparison-model]'))modelDraft[el.dataset.comparisonModel]=el.value;workflow();}update();};
    form.onclick=e=>{const b=e.target.closest('button');if(!b)return;
      if(b.dataset.selectCheckpoints){form.querySelectorAll('[name=checkpoint]').forEach(el=>el.checked=b.dataset.selectCheckpoints==='all');update();}
      if(b.hasAttribute('data-insert-trigger')){const field=form.elements.prompt;if(!field.value.includes(options.trigger))field.value=(field.value.trim()+' '+options.trigger).trim();update();}
      if(b.hasAttribute('data-random-seeds')){const n=Math.max(1,Math.min(16,read().seeds.length));form.elements.seeds.value=[...crypto.getRandomValues(new Uint32Array(n))].join(', ');update();}
    };
    form.onsubmit=async e=>{e.preventDefault();if(busy)return;const value=read();busy=true;update();$('#comparison-error').textContent='';
      try{const data=await api(`/jobs/${job}/comparisons`,'POST',value);$('#modal').close();await open(data.id);}
      catch(error){$('#comparison-error').textContent=error.message;}finally{busy=false;if(form.isConnected)update();}
    };
  }
  async function click(button,job,results){
    if(button.hasAttribute('data-compare-checkpoints')){await setup(job,results);return true;}
    if(button.dataset.comparisonOpen){await open(button.dataset.comparisonOpen);return true;}
    if(button.hasAttribute('data-comparison-back')){leave();await refreshResults();return true;}
    if(button.dataset.comparisonAction&&selected){button.disabled=true;try{draw(await api(`/comparisons/${selected.id}/${button.dataset.comparisonAction}`,'POST',{}));}finally{button.disabled=false;}return true;}
    if(button.hasAttribute('data-comparison-rerun')&&selected){button.disabled=true;try{const data=await api(`/comparisons/${selected.id}/rerun`,'POST',{});await open(data.id);}finally{button.disabled=false;}return true;}
    return false;
  }
  return {click,leave,visible:()=>Boolean(selected),render:()=>{if(selected){const d=selected;selected=null;draw(d);}}};
}
