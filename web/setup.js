import {escapeHTML as h} from './domain.js';
import {llmFields,llmSettings,captureLlm,openConfigEditor,openProfileManager} from './llm-settings.js';

const fields = ['toolkit_path', 'toolkit_python', ...llmFields];
const active = job => ['queued', 'running'].includes(job?.status);

export function setupWizard({api, state, onSaved, toast}) {
  const dialog = document.createElement('dialog');
  dialog.id = 'setup-wizard';
  dialog.setAttribute('aria-labelledby', 'setup-title');
  document.body.append(dialog);
  let profileList=[], modelList=[], manager=null, step = 0, draft, revision, status, report, busy = false, timer, vision = '', visionPassed = false, savedDraft, autoCheck, generation = 0;
  const capture = () => { for (const el of dialog.querySelectorAll('[name]')) if(fields.includes(el.name)&&!el.hasAttribute('data-llm-field'))draft[el.name] = el.type==='checkbox'?el.checked:el.value; captureLlm(dialog,draft); };
  const input = (name, label, hint = '') => `<label class="field">${label}<input name="${name}" value="${h(draft[name] || '')}" ${busy||active(status?.install)?'disabled':''}><small>${hint}</small></label>`;
  const row = (label, good, text) => `<div class="setup-check"><span class="badge ${good?'good':'warn'}">${good?'Ready':'Review'}</span><div><strong>${h(label)}</strong><p>${h(text)}</p></div></div>`;
  function checks() {
    if (!report) return '<p class="muted">Run checks to inspect this configuration. Existing settings are preserved until you save.</p>';
    const r = report.runtime, b = report.backend, c = report.connection, ps = c.promptstudio, local = c.standalone;
    const selected = local.available && (local.configured ?? local.models?.includes(draft.llm_model));
    return row('AI Toolkit', b.available, b.available ? `Supported architectures: ${b.architectures.join(', ') || 'none detected'}.` : 'Choose the folder containing run.py and its Python executable, or install a managed copy.')
      + row('Python & packages', r.available, r.reason)
      + (report.hardware?row('GPU driver', report.hardware.available, report.hardware.available?report.hardware.devices.map(d=>`${d.name} · ${d.vram_gb} GB · driver ${d.driver}`).join(', '):report.hardware.reason):'')
      + row('Training GPU', r.cuda, r.cuda ? r.devices.map(d => `${d.name} · ${d.vram_gb} GB VRAM`).join(', ') : !draft.toolkit_python ? 'Select a training Python environment to check CUDA. Dataset preparation remains available.' : 'CUDA is unavailable in this Python environment. Check the GPU driver and PyTorch build. You can still prepare datasets.')
      + row('LLM connection', ps.available || selected, visionPassed ? 'Synthetic image check passed. Caption accuracy still needs review.' : ps.available ? `Prompt Studio · ${ps.provider} · ${ps.vision?.available?'vision capability reported':'vision unavailable: '+(ps.vision?.reason || 'choose a vision model')}` : selected ? 'Selected standalone model is listed. Image support has not been verified.' : local.available ? 'Endpoint reached. Choose a listed model below.' : ps.reason || local.reason || 'Start a local LLM or connect Prompt Studio.')
      + (draft.caption_source !== 'connected' ? row('Dedicated captioner', c.captioner?.configured, c.captioner?.reason || 'Review the dedicated caption model in Settings.') : '')
      + row('Installation tools & storage', report.git && report.disk_free_gb >= 10, `${report.git?'Git found':'Git is needed for automatic installation'}. ${report.disk_free_gb} GB free in the data folder. Model weights and training outputs need additional space.`);
  }
  function render() {
    const job = status.install || {}, installing = active(job);
    let body;
    if (step === 0) body = `<span class="eyebrow">YOUR LOCAL TRAINING WORKSPACE</span><h3>A few checks before your first LoRA</h3><p class="muted">Connect the training backend and a local LLM. We’ll check what is available and help you fill in the gaps.</p><div class="setup-overview"><div><span>01</span><strong>Training backend</strong><p>Use an existing AI Toolkit installation or download a private copy.</p></div><div><span>02</span><strong>LLM & captions</strong><p>Reuse Prompt Studio’s connection or choose a standalone local endpoint. The Prompt Studio backend also enables direct checkpoint comparison.</p></div><div><span>03</span><strong>Review readiness</strong><p>Check the runtime, GPU and model support before training.</p></div></div><div class="note">You can prepare a dataset before training is ready. Reopen this wizard anytime from Settings → Run setup.</div>`;
    if (step === 1) body = `<h3>Connect AI Toolkit</h3><p class="muted">Already installed? Use the folder containing run.py and the Python executable from that installation.</p><div class="setup-fields">${input('toolkit_path','AI Toolkit folder')}${input('toolkit_python','AI Toolkit Python executable','Usually .venv/Scripts/python.exe on Windows or .venv/bin/python on Linux.')}</div><button data-setup="check" ${busy||installing?'disabled':''}>Save & check prerequisites</button><details class="setup-install" ${!draft.toolkit_path || installing?'open':''}><summary>Download and install AI Toolkit</summary><p>Downloads the official <a href="https://github.com/ostris/ai-toolkit" target="_blank" rel="noopener noreferrer">Ostris AI Toolkit</a> and runs its experimental installer manager. It chooses the Python and PyTorch build for your hardware. Downloads can be several GB and take a while.</p><p class="muted">Installs under <span class="setup-path">${h(status.managed_parent)}</span>. Existing installations are preserved. Models download later through AI Toolkit; some require license acceptance or a Hugging Face login in its environment.</p><div class="actions"><button data-setup="install" ${busy||installing?'disabled':''}>${['failed','cancelled','interrupted'].includes(job.status)?'Retry installation':'Download & install'}</button><a href="https://git-scm.com/downloads" target="_blank" rel="noopener noreferrer">Get Git</a><a href="https://github.com/ostris/ai-toolkit#installation" target="_blank" rel="noopener noreferrer">Installation guide</a></div></details><div id="setup-install-state" role="status">${job.status?`<div class="note"><strong>${h(job.phase)}</strong><p>${h(job.message || 'Installation continues if you close this wizard. Keep LoRA Studio running.')}</p>${installing?'<progress aria-label="Installing AI Toolkit"></progress><button data-setup="stop">Cancel installation</button>':''}${job.python?`<small class="setup-path">${h(job.python)}</small>`:''}</div>`:''}</div>${report?checks():''}`;
    if (step === 2) body = `<h3>Choose your LLM backend</h3>${llmSettings(draft,{models:modelList,profiles:profileList,manager,installation:status.promptstudio_install,disabled:busy||installing})}<div class="actions"><button data-setup="check" ${busy||installing?'disabled':''}>Save & check connections</button><button data-setup="vision" ${busy||installing?'disabled':''}>Test vision with a sample image</button></div><p class="muted">The vision test sends a generated color swatch. Dedicated captioners such as JoyCaption remain in Settings.</p><div role="status">${h(vision)}</div>${report?checks():''}`;
    if (step === 3) body = `<h3>${report?.training_ready?'Training runtime is available':'Review your setup'}</h3><p class="muted">${report?'These checks do not run training or verify model access, caption accuracy, or available memory for a particular model.':'Check prerequisites to see what is ready and what needs attention.'}</p>${checks()}<div class="note">Base-model downloads, licenses, and memory requirements depend on the model you choose. The Create step checks the selected architecture and dataset before launching AI Toolkit.</div><button data-setup="check" ${busy||installing?'disabled':''}>Save & run checks again</button>`;
    dialog.innerHTML = `<div class="modal-head"><div><span class="eyebrow">INITIAL SETUP</span><h2 id="setup-title">Welcome to LoRA Studio</h2></div><button data-setup="later" ${busy?'disabled':''} aria-label="Close setup">✕</button></div><nav class="setup-steps" aria-label="Setup steps">${['Welcome','AI Toolkit','LLM connection','Review'].map((name,i)=>`<button data-step="${i}" ${busy?'disabled':''} ${i===step?'aria-current="step"':''}><span>${i+1}</span>${name}</button>`).join('')}</nav><div class="setup-body">${body}</div><div id="setup-feedback" role="status">${busy?'Checking… This may take up to two minutes.':''}</div><div class="setup-footer"><button class="quiet" data-setup="later" ${busy?'disabled':''}>Set up later</button><div class="actions">${step?`<button data-setup="back" ${busy?'disabled':''}>Back</button>`:''}<button class="primary" data-setup="${step===3?'finish':'next'}" ${busy?'disabled':''}>${step===3?'Save & open workspace':'Continue'}</button></div></div>`;
  }
  async function save() {
    capture();
    const changes = fields.filter(k=>JSON.stringify(draft[k])!==JSON.stringify(savedDraft[k]));
    if (!changes.length) return state.settings;
    const submitted=Object.fromEntries(changes.map(k=>[k,draft[k]]));
    const saved = await api('/settings', 'PUT', {...submitted, settings_revision: revision});
    for(const key of changes)if(JSON.stringify(draft[key])===JSON.stringify(submitted[key]))draft[key]=saved[key];
    state.settings = saved; savedDraft={...saved}; revision = saved.settings_revision;
    return saved;
  }
  async function dismiss(disposition) {
    capture();
    // Keep edits when closing; validation/conflict failures keep the wizard open.
    if (!active(status.install)) await save();
    state.onboarding = await api('/setup/dismiss', 'POST', {disposition});
    dialog.close(); clearTimeout(timer); onSaved();
  }
  async function poll() {
    if (!dialog.open) return;
    try {
      const wasActive = active(status.install);
      const oldInstall = JSON.stringify(status.install);
      const oldManager=JSON.stringify(status.promptstudio_install);
      status = await api('/setup');
      const finished = wasActive && !active(status.install);
      if (finished) {
        const bootstrap = await api('/bootstrap');
        state.settings = bootstrap.settings; draft = {...state.settings}; savedDraft={...draft}; revision = draft.settings_revision;
        report = null;
      }
      if (!busy && (oldManager!==JSON.stringify(status.promptstudio_install)||oldInstall!==JSON.stringify(status.install) && (wasActive || active(status.install)))) { if(!finished)capture(); render(); }
    } catch (e) { dialog.querySelector('#setup-feedback').textContent = e.message; }
    if (dialog.open) timer = setTimeout(poll, 2000);
  }
  dialog.addEventListener('cancel', e=>{e.preventDefault();if(busy)return;dismiss('later').catch(e=>toast(e.message,true));});
  dialog.addEventListener('change', e=>{
    capture(); report = null; vision = ''; visionPassed=false;
    if(e.target.id==='setup-model') {draft.llm_model=e.target.value;render();}
    else if(e.target.hasAttribute('data-backend-choice')) {draft.integration=e.target.value;manager=null;modelList=[];render();}
    else if(e.target.name==='llm_provider'){modelList=[];delete draft._profileName;render();}
    else if(e.target.name==='prompt_studio_url'){manager=null;render();}
    else if(e.target.hasAttribute('data-mode-list')){render();}
    else if(e.target.hasAttribute('data-llm-saved-profile')){
      const selected=e.target.value, provider=draft.llm_provider;
      draft.llm_profile_selection={...draft.llm_profile_selection,[provider]:selected};
      if(selected)draft.llm_profiles={...draft.llm_profiles,[provider]:structuredClone(draft.llm_profile_library[selected])};
      delete draft._profileName;render();
    }
  });
  dialog.addEventListener('click', async e=>{
    const b=e.target.closest('button'); if(!b)return;
    e.stopPropagation();
    if(b.dataset.step!==undefined) {capture();step=Number(b.dataset.step);render();return;}
    if(b.dataset.llmAction){await llmAction(b.dataset.llmAction);return;}
    const action=b.dataset.setup; if(!action)return;
    try {
      capture();
      if(action==='later') {await dismiss('later');return;}
      if(action==='back'||action==='next'){step+=action==='next'?1:-1;render();return;}
      if(action==='finish'){await dismiss('complete');return;}
      if(action==='stop'){await api('/setup/stop','POST',{});b.disabled=true;b.textContent='Stopping…';return;}
      busy=true;render();
      if(autoCheck) await autoCheck;
      await save();
      if(action==='check'){report=await api('/setup/check','POST',{});modelList=report.connection.standalone.models||[];}
      if(action==='vision') {const result=await api('/setup/vision','POST',{});vision=result.reason;visionPassed=result.passed;}
      if(action==='install'){status=await api('/setup/install','POST',{});report=null;}
      busy=false;render();
    } catch(err){busy=false;render();dialog.querySelector('#setup-feedback').textContent=err.message;}
  });
  async function llmAction(action) {
    try {
      capture();
      if(action==='save-llm-profile'||action==='delete-llm-profile'){
        const provider=draft.llm_provider, selected=draft.llm_profile_selection?.[provider]||'';
        const library={...draft.llm_profile_library};
        if(action==='save-llm-profile'){
          const name=(draft._profileName||'').trim();if(!name)throw new Error('Enter a name for this LLM profile.');
          const id=library[selected]?.name===name?selected:crypto.randomUUID();
          library[id]={...draft.llm_profiles[provider],name};draft.llm_profile_selection={...draft.llm_profile_selection,[provider]:id};
        }else{
          delete library[selected];draft.llm_profile_selection=Object.fromEntries(Object.entries(draft.llm_profile_selection).map(([p,id])=>[p,id===selected?'':id]));
          delete draft._profileName;
        }
        draft.llm_profile_library=library;await save();render();return;
      }
      if(['new-profile','edit-profile'].includes(action)) {
        openConfigEditor(document.body,draft.llamacpp_config_profile,action==='new-profile',async result=>{
          draft.llamacpp_config_profile=result.config_profile;
          profileList=(await api('/llm/profiles')).profiles;render();
        });return;
      }
      if(action==='manage-profiles') {
        openProfileManager(document.body,async result=>{
          if(draft.llamacpp_config_profile===result.deleted){draft.llamacpp_config_profile='';draft.llamacpp_autostart=false;}
          const latest=await api('/bootstrap');state.settings=latest.settings;savedDraft={...latest.settings};revision=latest.settings.settings_revision;
          profileList=(await api('/llm/profiles')).profiles;render();
        });return;
      }
      busy=true;render();if(autoCheck)await autoCheck;
      await save();let message='';
      if(action==='refresh-profiles'){profileList=(await api('/llm/profiles')).profiles;message='Profiles refreshed. Your selected profile is preserved.';}
      if(action==='models'){const result=await api('/connection');modelList=result.standalone.models||[];message=result.standalone.available?`${modelList.length} model(s) found. ${result.standalone.vision?.available?'Vision capability reported.':result.standalone.vision?.reason||'Choose a model and test vision.'}`:result.standalone.reason;}
      if(action==='detect-manager'){manager=await api('/setup/promptstudio');}
      if(action==='install-promptstudio'){status=await api('/setup/promptstudio/install','POST',{acknowledged:true,expected_url:manager?.url});message=status.promptstudio_install.message;}
      if(action==='browse-executable'){const result=await api('/llm/pick-file','POST',{kind:'executable',current_path:draft.llamacpp_executable});if(result.path)draft.llamacpp_executable=result.path;}
      if(action.startsWith('server-')){const result=await api('/llm/server/'+action.slice(7),'POST',{});message=result.startup_error|| (result.running?`Server running${result.managed?' · managed by LoRA Studio':' · external server'}.`: 'Managed server is stopped.');}
      busy=false;render();dialog.querySelector('#setup-feedback').textContent=message||'';
    }catch(error){busy=false;render();dialog.querySelector('#setup-feedback').textContent=error.message;}
  }
  return async function open(startStep=0) {
    clearTimeout(timer);
    const bootstrap=await api('/bootstrap');
    state.settings=bootstrap.settings; draft={...state.settings}; savedDraft={...draft}; revision=draft.settings_revision;
    status=bootstrap.onboarding; report=null; step=active(status.install)?1:startStep;profileList=[];modelList=[];manager=null; busy=false;vision='';visionPassed=false;autoCheck=null;
    render();if(!dialog.open)dialog.showModal();timer=setTimeout(poll,2000);
    const currentGeneration=++generation, initial={...draft};
    api('/llm/profiles').then(result=>{if(currentGeneration===generation&&dialog.open)profileList=result.profiles;}).catch(()=>{});
    if(!active(status.install)) {
      dialog.querySelector('#setup-feedback').textContent='Checking saved prerequisites in the background…';
      autoCheck=api('/setup/check','POST',{}).then(result=>{
        if(currentGeneration!==generation||!dialog.open)return;
        capture();
        if(fields.every(k=>JSON.stringify(draft[k])===JSON.stringify(initial[k]))){report=result;modelList=result.connection.standalone.models||[];}
        if(!busy)dialog.querySelector('#setup-feedback').textContent=report?'Initial checks finished. Continue to review and configure your connections.':'';
      }).catch(error=>{if(currentGeneration===generation&&dialog.open&&!busy)dialog.querySelector('#setup-feedback').textContent=error.message;});
    }
  };
}
