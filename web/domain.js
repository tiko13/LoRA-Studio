export function escapeHTML(value) { return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
export function imageState(image, digest) {
  if (!image.included) return {label:'Excluded', tone:''};
  if (image.object_selection && image.object_selection.status!=='ready') return {label:'Object selection needed', tone:'warn'};
  if (!image.caption) return {label:'Needs caption', tone:'warn'};
  if (image.intent_digest !== digest) return {label:'Rules changed', tone:'warn'};
  if (image.preparation_stale) return {label:'Crop review needed', tone:'warn'};
  if (image.visual_review?.status==='concerns') return {label:'Visual concerns', tone:'warn'};
  if (image.warnings?.length && !image.reviewed) return {label:'Review needed', tone:'warn'};
  if (!image.manual && !image.reviewed) return {label:'Caption draft', tone:'warn'};
  return {label:image.manual ? 'Manual · ready' : 'Ready', tone:'good'};
}
export function resolvedAttributes(project, catalog) {
  const kept=project.kind==='concept'?catalog.concept_families?.[project.concept_family||'general']?.keep:null;
  return catalog.types.find(t => t.id === project.kind).attributes.map(a => {
    const preset=kept?(kept.includes(a.id)?'keep':'change'):a.default;
    return {...a, default:preset, resolved:!project.choices[a.id]||project.choices[a.id]==='auto'?preset:project.choices[a.id]};
  });
}
