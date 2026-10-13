import {escapeHTML as h} from './domain.js';

export const captionSourceLabel = source => ({connected:'Connected local LLM',joycaption:'JoyCaption Beta One',custom:'Custom caption model'}[source] || 'Connected local LLM');

export function captionSettings(s) {
  const source=s.caption_source||'connected';
  const input=(label,name,value,help='',type='text')=>`<label class="field full">${label}<input name="${name}" type="${type}" value="${h(value)}" ${type==='number'?'min="0" max="15" step="1"':''}><small>${help}</small></label>`;
  return `<h3>Captioning</h3><label class="field">Read images with<select name="caption_source">${['connected','joycaption','custom'].map(key=>`<option value="${key}" ${key===source?'selected':''}>${captionSourceLabel(key)}</option>`).join('')}</select><small>All options run locally. A dedicated captioner reads the image; the connected LLM below applies your learning rules to its observations. New captions always remain reviewable drafts.</small></label>
    <div data-caption-options="joycaption" ${source==='joycaption'?'':'hidden'}><details><summary>JoyCaption local runtime</summary><p class="muted">Studio starts a temporary llama.cpp server for this caption batch and stops it before preparing the final text. Configure Beta One weights and their matching vision projector. No automatic model downloads.</p><div class="form-grid">${input('llama.cpp executable','caption_runtime',s.caption_runtime||'','Full path to llama-server or llama.')}${input('JoyCaption model GGUF','joycaption_model',s.joycaption_model||'')}${input('Matching vision projector GGUF','joycaption_projector',s.joycaption_projector||'')}${input('Caption GPU index','caption_gpu',s.caption_gpu??0,'Idle ComfyUI model caches are released when connected. Active generations are never interrupted.','number')}</div></details></div>
    <div data-caption-options="custom" ${source==='custom'?'':'hidden'}><div class="form-grid">${input('Caption endpoint base URL','caption_url',s.caption_url||'http://127.0.0.1:8091/v1','Local OpenAI-compatible endpoint that accepts images.')}${input('Caption model ID','caption_model',s.caption_model||'','Exact model ID, for example a V4.5, Gliese or another caption model served by your runtime.')}</div></div><div class="divider"></div>`;
}

export function updateCaptionSettings(form) {
  const source=form.elements.caption_source.value;
  for(const group of form.querySelectorAll('[data-caption-options]')) group.hidden=group.dataset.captionOptions!==source;
}
