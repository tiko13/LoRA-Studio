export const preparationLocked = project => Boolean(project?.preparation_lock);
export function newProjectAction({state, save, api, selectProject, toast}) {
  let pending = false;
  return async () => {
    if (pending) return;
    pending = true;
    try {
      await save();
      let project, reused = false;
      try {
        project = await api('/projects', 'POST', {selected_project_id: state.current?.project.id});
      } catch (error) {
        if (error.code !== 'empty_project_exists' || !error.project) throw error;
        project = error.project;
        if (project.id === state.current?.project.id) {
          toast('An empty LoRA project already exists.', true);
          return;
        }
        reused = true;
      }
      state.projects = [project, ...state.projects.filter(p => p.id !== project.id)];
      state.tab = 'setup';
      await selectProject(project.id);
      if (reused) toast('An empty LoRA project already exists, switching to it.', false, 'success');
    } finally {
      pending = false;
    }
  };
}
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
