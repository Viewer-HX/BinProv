"""Regression checks for archived-parameter replay; no GPU or real corpus needed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = module('run_best')


class BestRecipes(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.load_config()

    def test_exact_members_and_no_duplicate_training(self):
        groups = self.cfg['groups']
        expected = ['r4_ctx2048_dense'] + [f'r6_dense2048_seed{s}' for s in (7, 13, 29)] + [f'r7_wide2048_seed{s}' for s in (7, 13, 29)]
        self.assertEqual(groups['o2o3_7wide_512B']['runs'], expected)
        steps = runner.plan(self.cfg, [k for k, g in groups.items() if g['kind'] != 'archived_no_recipe'], 'all')
        self.assertEqual(len([s for s in steps if s['phase'] == 'train']), 13)
        self.assertEqual(len([s for s in steps if s['phase'] == 'pretrain']), 2)
        self.assertEqual(steps[0]['phase'], 'prepare')

    def test_generated_flags_parse_and_match_all_archived_values(self):
        parsers = {n: module(n) for n in ['experiment', 'pretrain_mlm']}
        steps = runner.plan(self.cfg, ['o2o3_7wide_512B', 'opt4_6run_16.9KB'], 'all')
        for step in steps:
            if step['phase'] not in ('train', 'pretrain'):
                continue
            cmd = step['cmd']
            parser = parsers[Path(cmd[1]).stem]
            with patch.object(sys, 'argv', cmd[1:]):
                parsed = vars(parser.parse_args())
            if step['phase'] == 'train':
                original = runner.read_json(self.cfg['runs'][step['name']]['archived_args'])
            elif step['name'] == 'mlm512':
                original = runner.read_json(self.cfg['prerequisites']['mlm512']['archived_args'])
            else:
                self.assertEqual(parsed['seq_bytes'], 2048)
                self.assertTrue(parsed['init_from'].endswith('mlm512'))
                continue
            for key, value in original.items():
                if key not in ('corpus', 'out', 'init_from'):
                    self.assertEqual(parsed[key], value, (step['name'], key))
            if step['name'].startswith(('r4_', 'r6_')):
                self.assertNotIn('--val-seed', cmd)
                self.assertIsNone(parsed['val_seed'])

    def test_dry_run_has_no_subprocess_or_writes(self):
        with patch.object(runner.subprocess, 'call', side_effect=AssertionError('executed')), patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir')), patch.object(Path, 'write_text', side_effect=AssertionError('write')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--group', 'all', '--dry-run']), 0)
            self.assertEqual(runner.main(['--group', 'opt4_3wide_binary', '--phase', 'offline', '--archive-probs']), 0)

    def test_checkpoint_config_without_weights_is_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'binprov_config.json').write_text('{}')
            self.assertFalse(runner.checkpoint_ready(p))
            (p / 'model.safetensors').write_bytes(b'fixture')
            self.assertTrue(runner.checkpoint_ready(p))

    def test_missing_corpus_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner.subprocess, 'call', side_effect=AssertionError('executed')), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(runner.main(['--group', 'opt4_3wide_binary', '--phase', 'train', '--execute', '--corpus', d]), 2)
            self.assertIn('build the corpus', err.getvalue())

    def test_unavailable_recipe_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(['--group', 'o2o3_19run_binary']), 2)

    def test_partition_check_detects_program_leakage(self):
        from types import SimpleNamespace
        import numpy as np
        groups = ['a', 'b', 'c']
        np.random.default_rng(1234).shuffle(groups)
        corpus = SimpleNamespace(records=[SimpleNamespace(group=g) for g in ['a', 'b', 'c', 'test']],
                                 load_splits=lambda name: {'train': [0, 1, 2], 'test': [3]})
        with tempfile.TemporaryDirectory() as d:
            evidence = Path(d) / 'split.txt'
            evidence.write_text('[TEST]\ntest\n[VALIDATION]\n' + groups[0] + '\n')
            self.cfg['split_evidence'] = str(evidence)
            with patch('binprov.corpus.Corpus', return_value=corpus), contextlib.redirect_stdout(io.StringIO()):
                runner.verify_corpus(self.cfg)
                corpus.records[0].group = 'test'
                with self.assertRaisesRegex(ValueError, 'SPLIT MISMATCH'):
                    runner.verify_corpus(self.cfg)

    def test_archive_metadata_precedence_and_staging(self):
        cfg = self.cfg
        cfg['archive_probs'] = True
        tag = cfg['groups']['opt4_3wide_binary']['runs'][0]
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'ROOT', Path(d)):
            archive = Path(d) / cfg['runs'][tag]['archived_result']
            archive.parent.mkdir(parents=True)
            archive.write_text('{"test":{"accuracy":0.5}}')
            self.assertEqual(runner.offline_result(cfg, tag), archive)
            local = runner.offline_dir(cfg, tag)
            local.mkdir(parents=True)
            (local / 'result.json').write_text('{"test":{"accuracy":0.6}}')
            (local / 'probs.npz').write_bytes(b'fixture')
            step = {'name': 'fixture', 'title': 'fixture', 'runs': [tag], 'cmd': ['unused'], 'outputs': ['results/best/tables/fixture.md']}
            self.assertEqual(runner.offline_result(cfg, tag), local / 'result.json')
            with patch.object(runner.subprocess, 'call', return_value=0), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_offline(step, cfg, True), 0)
            staged = Path(d) / cfg['results_dir'] / 'archive_inputs' / tag
            self.assertEqual((staged / 'result.json').read_bytes(), (local / 'result.json').read_bytes())
            self.assertEqual((staged / 'probs.npz').read_bytes(), b'fixture')


if __name__ == '__main__':
    unittest.main()
