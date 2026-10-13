import {escapeHTML as h} from './domain.js';

export function trainingLabel(job) {
  if(job.status==='paused')return 'Training paused';
  if(job.status==='stopping')return 'Stopping training';
  if(job.status==='queued')return 'Training queued';
  if(job.status==='running')return job.pid&&job.phase==='Training'?'LoRA in training':'Preparing for training';
  return '';
}
export function visibleTraining(jobs) {
  return jobs.filter(j=>j.kind==='train'&&trainingLabel(j)).sort((a,b)=>
    Number(a.status==='paused')-Number(b.status==='paused')||b.created-a.created);
}
export function projectTrainingBadge(jobs, project) {
  const runs=jobs.filter(j=>j.kind==='train'&&j.project===project);
  const active=visibleTraining(runs)[0];
  if(active)return `<span class="project-training-badge">${h(trainingLabel(active))}</span>`;
  const latest=runs.sort((a,b)=>b.created-a.created)[0];
  return latest?.status==='complete'?'<span class="project-training-badge complete">✓ Training complete</span>':'';
}
function stepProgress(job, progress) {
  if(job.status!=='running'||!job.pid||job.phase!=='Training')return '';
  if(!Number.isFinite(progress?.step)||progress.step<0||!Number.isFinite(progress?.total)||progress.total<=0)return '<small>Waiting for step progress…</small>';
  const step=Math.min(progress.step,progress.total),percent=100*step/progress.total;
  return `<span class="training-step-progress"><span>Step ${h(step.toLocaleString())} / ${h(progress.total.toLocaleString())} · ${h(percent.toLocaleString(undefined,{maximumFractionDigits:1}))}%</span><progress value="${step}" max="${progress.total}" aria-label="Training steps"></progress></span>`;
}
export function activityCards(jobs, projects, progressByJob=new Map()) {
  return visibleTraining(jobs).map(j=>`<button class="training-activity-card ${j.status==='paused'?'paused':''}" data-dashboard-job="${h(j.id)}" data-dashboard-project="${h(j.project)}"><span class="training-activity-label">${h(trainingLabel(j))}</span><strong>${h(projects.find(p=>p.id===j.project)?.name||'LoRA project')}</strong>${stepProgress(j,progressByJob.get(j.id))}<small>Open dashboard →</small></button>`).join('');
}
