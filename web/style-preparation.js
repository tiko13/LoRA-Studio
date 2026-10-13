import {escapeHTML as h} from './domain.js';

export function styleSettings(project,catalog) {
  if(project.kind!=='style') return '';
  const family=project.style_family||'general', families=catalog.style_families||{};
  return `<details class="full optional-settings" id="style-family-options"><summary>Optional · style family</summary><label class="field">Style family<select data-field="style_family">${Object.entries(families).map(([id,p])=>`<option value="${h(id)}" ${family===id?'selected':''}>${h(p.label)}</option>`).join('')}</select><small>${h(families[family]?.note||'')}</small></label><p class="muted">This guides captioning; it does not change your images or assert that they match the style. Keep varied subjects with one coherent look.</p></details>`;
}

export function styleSummary(project) {
  if(project.kind!=='style') return '';
  return `<p class="muted">Style preparation describes subjects, actions and setting while the trigger carries the look. Palette, texture, linework, medium and print borders stay implicit by default; object colours remain captionable. Adjust these choices under learning rules.</p><p class="muted">Review the training preview for lost borders, fine strokes or pixel edges. Minimal cropping preserves proportions; use full-image padding only when needed. Padding and repeated signatures can become part of the learned style.</p>`;
}
