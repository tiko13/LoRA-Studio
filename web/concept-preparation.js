import {escapeHTML as h} from './domain.js';

export function conceptSettings(project,catalog) {
  if(project.kind!=='concept') return '';
  const selected=project.concept_family||'general', families=catalog.concept_families||{};
  return `<label class="field full">Concept type<select data-field="concept_family">${Object.entries(families).map(([id,p])=>`<option value="${h(id)}" ${selected===id?'selected':''}>${h(p.label)}</option>`).join('')}</select><small>${h(families[selected]?.note||'')} This adjusts Auto learning rules; your explicit Keep/Change choices are preserved.</small></label>`;
}

export function conceptSummary(project,catalog) {
  if(project.kind!=='concept') return '';
  return `<p class="muted">${h(catalog.concept_families?.[project.concept_family||'general']?.note||'')} The trigger carries the selected concept while captions describe visible variable context.</p><p class="muted">Check the training crop for the evidence that matters: face, joints, contact points or scene layout. Concept checks are advisory; an uncertain or missing target stays available for your review.</p>`;
}
