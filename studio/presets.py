"""Portable training choices extracted from a frozen run; saved locally, never in source."""
import hashlib
import json

from .domain import normalize_project, bounded_text


def from_run(name, config, manifest, *, default=False):
    process = config['config']['process'][0]
    project = manifest['project']
    train, sample, save = process['train'], process['sample'], process['save']
    training = {k: v for k, v in project['training'].items() if k != 'gpu'}
    training.update(steps=train['steps'], lr=train['lr'], optimizer=train['optimizer'],
                    rank=process['network']['linear'], resolution=process['datasets'][0]['resolution'][0],
                    seed=sample['seed'], checkpoint_every=save['save_every'],
                    sample_every=0 if train.get('disable_sampling') else sample['sample_every'],
                    keep_checkpoints=save['max_step_saves_to_keep'], sample_steps=sample['sample_steps'])
    normalized = normalize_project({'model': project['model'], 'training': training})['training']
    normalized.pop('gpu')
    return {'name': bounded_text(name, 120, 'Preset name'), 'model': project['model'],
            'training': normalized, 'preparation': project.get('preparation', {}), 'default': default,
            'source': {'run_id': manifest['run_id'], 'backend_commit': manifest.get('backend_commit'),
                       'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()},
            'note': 'Baseline from an accepted run. Starting point for this model, not a universal quality optimum.'}
