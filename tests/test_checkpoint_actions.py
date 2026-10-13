"""Destructive result actions tested only against disposable synthetic runs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from studio.checkpoint_actions import delete_checkpoint
from studio.comparisons import Comparisons
from studio.service import Service
from studio.store import Conflict
from tests.test_results import TEST_ROOT, fixture, weights


class CheckpointDeletionCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(self.temp.name)
        self.service.comparisons = Comparisons(self.service, 'http://127.0.0.1:8787')
        self.addCleanup(self.service.comparisons.close)
        self.job, self.root, self.output = fixture(self.service.store)
        self.path = self.output / 'lora_results-run_000000025.safetensors'
        self.name = self.path.relative_to(self.root).as_posix()
        weights(self.path, 25)

    def delete(self):
        plan = delete_checkpoint(self.service, self.job['id'], self.name)
        return delete_checkpoint(self.service, self.job['id'], self.name, plan['token'])

    def comparison(self, status='complete', selected=True):
        identity = uuid.uuid4().hex
        root = self.service.comparisons.directory(identity)
        (root / 'weights').mkdir(parents=True)
        (root / 'images').mkdir()
        (root / 'weights/snapshot.safetensors').write_bytes(b'synthetic retained weights')
        (root / 'images/image.png').write_bytes(b'synthetic image')
        (root / 'grid.png').write_bytes(b'synthetic grid')
        self.service.comparisons.save({'id': identity, 'job': self.job['id'], 'status': status,
            'created': 1, 'columns': [{'path': self.name if selected else 'output/other.safetensors'}]})
        return root

    def test_removes_related_artifacts_preserves_other_checkpoints_and_inputs(self):
        final = self.output / 'lora_results-run.safetensors'
        weights(final, 100)
        optimizer = self.output / 'optimizer.pt'
        optimizer.write_bytes(b'synthetic latest optimizer')
        sidecar = self.path.with_suffix('.yaml')
        sidecar.write_text('synthetic metadata')
        preview = self.output / 'samples/123__000000025_0.png'
        preview.write_bytes(b'synthetic preview')
        (preview.parent / '.thumbs').mkdir()
        thumb = preview.parent / '.thumbs' / (preview.stem + '.jpg')
        thumb.write_bytes(b'synthetic thumbnail')
        baseline = preview.parent / '123__000000000_0.png'
        baseline.write_bytes(b'baseline')
        original = self.root / 'originals/source.png'
        original.parent.mkdir()
        original.write_bytes(b'synthetic original')
        related, unrelated = self.comparison(), self.comparison(selected=False)
        # Match the backend selection contract without relying on timestamp resolution.
        with patch('studio.checkpoint_actions.latest_checkpoint', return_value=final):
            plan = delete_checkpoint(self.service, self.job['id'], self.name)
            self.assertTrue(self.path.exists())
            self.assertEqual(plan['comparisons'], 1)
            self.assertFalse(plan['optimizer'])
            result = delete_checkpoint(self.service, self.job['id'], self.name, plan['token'])
        self.assertTrue(result['deleted'])
        for deleted in (self.path, sidecar, preview, thumb, related):
            self.assertFalse(deleted.exists(), deleted.name)
        for kept in (final, optimizer, baseline, original, unrelated, self.root / 'config.json'):
            self.assertTrue(kept.exists(), kept.name)
        self.assertEqual(self.service.store.item('jobs', self.job['id']), self.job)

    def test_latest_optimizer_and_snr_removed_but_shared_previews_remain(self):
        duplicate = self.output / 'lora_results-run.safetensors'
        weights(duplicate, 25)
        preview = self.output / 'samples/123__000000025_0.png'
        preview.write_bytes(b'synthetic preview')
        for filename in ('optimizer.pt', 'learnable_snr.json'):
            (self.output / filename).write_bytes(b'synthetic')
        with patch('studio.checkpoint_actions.latest_checkpoint', return_value=self.path):
            result = self.delete()
        self.assertTrue(result['shared_previews'])
        self.assertTrue(result['optimizer'])
        self.assertTrue(preview.exists())
        self.assertTrue(duplicate.exists())
        self.assertFalse((self.output / 'optimizer.pt').exists())
        self.assertFalse((self.output / 'learnable_snr.json').exists())

    def test_changed_artifacts_require_new_confirmation(self):
        plan = delete_checkpoint(self.service, self.job['id'], self.name)
        self.path.with_suffix('.yaml').write_text('new sidecar')
        with self.assertRaises(Conflict):
            delete_checkpoint(self.service, self.job['id'], self.name, plan['token'])
        self.assertTrue(self.path.exists())

    def test_incomplete_newer_checkpoint_does_not_own_previous_optimizer(self):
        broken = self.output / 'lora_results-run_000000050.safetensors'
        broken.write_bytes(b'partial later save')
        optimizer = self.output / 'optimizer.pt'
        optimizer.write_bytes(b'previous completed optimizer')
        name = broken.relative_to(self.root).as_posix()
        plan = delete_checkpoint(self.service, self.job['id'], name)
        self.assertFalse(plan['optimizer'])
        delete_checkpoint(self.service, self.job['id'], name, plan['token'])
        self.assertTrue(optimizer.exists())
        self.assertTrue(self.path.exists())

    def test_active_unresolved_and_live_processes_are_blocked(self):
        for status in ('queued', 'running', 'stopping', 'interrupted'):
            self.job['status'] = status
            self.service.store.put_item('jobs', self.job)
            with self.subTest(status=status), self.assertRaises(Conflict):
                self.delete()
        self.job.update(status='complete', pid=123)
        self.service.store.put_item('jobs', self.job)
        with patch('studio.checkpoint_actions.running', return_value=True), self.assertRaises(Conflict):
            self.delete()
        self.assertTrue(self.path.exists())

    def test_reservation_and_attached_owner_block_deletion(self):
        self.service.job_lock.acquire()
        try:
            with self.assertRaises(Conflict):
                self.delete()
        finally:
            self.service.job_lock.release()
        self.service.owner_url = 'http://127.0.0.1:9999'
        with self.assertRaises(Conflict):
            self.delete()

    def test_interrupted_comparison_is_preserved(self):
        comparison = self.comparison(status='interrupted')
        with self.assertRaises(Conflict):
            self.delete()
        self.assertTrue(comparison.exists())
        self.assertTrue(self.path.exists())

    def test_move_failure_rolls_back_every_file(self):
        preview = self.output / 'samples/123__000000025_0.png'
        preview.write_bytes(b'synthetic preview')
        replace = Path.replace
        calls = 0
        def fail_second(path, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('Synthetic sharing violation')
            return replace(path, target)
        with patch.object(Path, 'replace', fail_second), self.assertRaises(OSError):
            self.delete()
        self.assertTrue(self.path.exists())
        self.assertTrue(preview.exists())

    def test_paths_incomplete_weights_and_missing_files(self):
        for name in ('../config.json', 'output/../config.json', self.path.as_posix(),
                     'output/samples/test.png', 'output/missing.safetensors'):
            with self.subTest(name=name), self.assertRaises((ValueError, KeyError)):
                delete_checkpoint(self.service, self.job['id'], name)
        self.path.write_bytes(b'partial checkpoint')
        self.assertTrue(self.delete()['deleted'])
        with self.assertRaises(KeyError):
            self.delete()

    def test_symlink_sidecar_cannot_delete_another_file(self):
        outside = self.root / 'private.json'
        outside.write_text('synthetic')
        try:
            self.path.with_suffix('.json').symlink_to(outside)
        except OSError:
            self.skipTest('Symlink creation unavailable')
        with self.assertRaises(ValueError):
            self.delete()
        self.assertTrue(outside.exists())
        self.assertTrue(self.path.exists())

    def test_only_selected_continuation_folder_is_cleaned(self):
        continued = self.root / 'output/continuation-1/lora_results-run'
        (continued / 'samples').mkdir(parents=True)
        weights(continued / self.path.name, 25)
        (continued / 'optimizer.pt').write_bytes(b'new optimizer')
        (continued / 'samples/123__000000025_0.png').write_bytes(b'new preview')
        self.delete()
        self.assertEqual(len(list(continued.rglob('*'))), 4)
