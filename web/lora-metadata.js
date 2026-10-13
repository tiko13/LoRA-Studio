import {escapeHTML as h} from './domain.js';

export function metadataPanel(project) {
  const m=project.lora_metadata||{};
  return `<section class="card"><details><summary>LoRA file metadata · optional</summary>
    <p class="muted">These fields travel inside every generated LoRA and checkpoint, together with the LoRA name and training settings. Changes apply to future runs.</p>
    <div class="form-grid extra-rules">
      <label class="field full">Public description<textarea data-lora-metadata="description" maxlength="4000" placeholder="Describe the LoRA for someone using it.">${h(m.description||'')}</textarea><small>Separate from your private learning-target description.</small></label>
      <label class="field">Author<input data-lora-metadata="author" maxlength="200" value="${h(m.author||'')}" placeholder="Name or alias"></label>
      <label class="field">License<input data-lora-metadata="license" maxlength="500" value="${h(m.license||'')}" placeholder="License identifier, URL or short terms"><small>Blank means unspecified.</small></label>
      <label class="field">Trigger words<textarea data-lora-metadata="trigger_words" placeholder="${h(project.trigger||'Use the training trigger')}">${h((m.trigger_words||[]).join('\n'))}</textarea><small>One phrase per line. Blank follows the training trigger. This describes usage; it does not change captions or the training trigger.</small></label>
      <label class="field">Recommended strength<input data-lora-metadata="recommended_strength" type="number" min="-10" max="10" step="any" value="${h(m.recommended_strength??'')}" placeholder="Unspecified"><small>Optional suggestion for inference, from −10 to 10. No automatic strength is inferred.</small></label>
    </div><p class="muted"><small>Private learning instructions, captions, preview prompts and local paths are not copied into file metadata. Enter only information you want to share in these public fields.</small></p>
  </details></section>`;
}

export function setMetadataField(project, key, text) {
  if(!['description','author','license','trigger_words','recommended_strength'].includes(key))return false;
  let value=text;
  if(key==='trigger_words')value=text.trim()?text.split(/\r?\n/):null;
  if(key==='recommended_strength')value=text.trim()===''?null:Number(text);
  project.lora_metadata={...(project.lora_metadata||{}),[key]:value};
  return true;
}
