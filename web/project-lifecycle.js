export const preparationLocked = project => Boolean(project?.preparation_lock);
export function lockNotice(project) {
  if (!preparationLocked(project)) return '';
  return `<div class="note preparation-notice"><div><strong>${project.preparation_lock.started?'Preparation locked':'Training is starting'}</strong><p>${project.preparation_lock.started?'This run has started. Its base info, dataset, captions and training settings are fixed, including after it stops or finishes.':'Frozen preparation is held while the trainer starts. If it fails before its first training step, preparation becomes editable again.'}</p><p>Copy all preparation into a fresh, editable project to start another run.</p></div><button data-action="copy-run">Copy to new run</button></div>`;
}
export function disablePreparation(container) {
  container.querySelectorAll('input,textarea,select,button').forEach(control => {
    if (control.id === 'dataset-filter' || control.name === 'show_boxes' ||
        control.matches('[data-image],[data-goto],[data-action="settings"],[data-action="discard"],[data-action="download-draft"],[data-action="copy-run"]')) return;
    if (!control.disabled) control.dataset.preparationDisabled = 'true';
    control.disabled = true;
  });
}
export function enablePreparation(container) {
  container?.querySelectorAll('[data-preparation-disabled]').forEach(control => {
    control.disabled = false;
    delete control.dataset.preparationDisabled;
  });
}
