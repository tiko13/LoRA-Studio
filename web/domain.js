export function escapeHTML(value) { return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
export function imageState(image, digest) {
  if (!image.included) return {label:'Excluded', tone:''};
  if (!image.caption) return {label:'Needs caption', tone:'warn'};
  if (image.intent_digest !== digest) return {label:'Rules changed', tone:'warn'};
  if (image.warnings?.length && !image.reviewed) return {label:'Review needed', tone:'warn'};
  return {label:image.manual ? 'Manual · ready' : 'Ready', tone:'good'};
}
export function resolvedAttributes(project, catalog) {
  return catalog.types.find(t => t.id === project.kind).attributes.map(a => ({...a, resolved: !project.choices[a.id] || project.choices[a.id] === 'auto' ? a.default : project.choices[a.id]}));
}
