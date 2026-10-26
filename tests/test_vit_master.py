"""Synthetic CPU checks for the self-contained master notebook; no dataset/CLIP download.

Run from the repository root with its normal notebook dependencies:
    python -m unittest discover -s tests -p 'test_vit_master.py' -v
The encoder below is a toy fixture, not a substitute for a CARC/CLIP smoke run.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

NOTEBOOK = Path(__file__).resolve().parents[1] / 'ViT_Master_Experiments.ipynb'


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode_image(self, images):
        self.calls += 1
        return torch.cat([images.mean((2, 3)), images.std((2, 3))], dim=1)


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.nb = json.loads(NOTEBOOK.read_text())
        # Execute the actual notebook definitions from a real file so inspect works.
        source = ['import sys, types\nsys.modules.setdefault("clip", types.ModuleType("clip"))']
        for cell in cls.nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            text = ''.join(cell['source'])
            if text.startswith(('validate_settings()', 'session_rows =', 'rows, missing =')):
                continue
            if text.startswith('import contextlib'):
                source.append('REQUIRE_GPU = False')
                text = text.replace('from IPython.display import display', 'display = lambda value: None')
            source.append(text)
        py = cls.root / 'notebook_definitions.py'
        py.write_text('\n\n'.join(source))
        spec = importlib.util.spec_from_file_location('vit_master_under_test', py)
        cls.m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.m
        spec.loader.exec_module(cls.m)
        m = cls.m
        cls.default_plan_size = len(m.HOLDOUTS)*len(m.K_VALUES)*len(m.REPEATS)*len(m.SEEDS)*len(m.TRAINING_MODES)
        m.DATA_ROOT = cls.root / 'GenImage_Compact'
        m.DATA_ROOT.mkdir()
        m.MANIFEST_ROOT = m.DATA_ROOT / '_experiment_manifests'
        m.MANIFEST_ROOT.mkdir()
        m.OUTPUT_ROOT = cls.root / 'results'
        m.FEATURE_CACHE_ROOT = cls.root / 'features'
        m.MAX_EPOCHS = 2
        m.PATIENCE = 3
        m.IMAGE_BATCH_SIZE = 4
        m.HEAD_BATCH_SIZE = 4
        m.EXPECTED_COUNTS = {'train': {0: 4, 1: 4}, 'val': {0: 2, 1: 2}, 'test': {0: 2, 1: 2}}
        m.ENCODER = ToyEncoder()
        m.PREPROCESS = transforms.Compose([transforms.Resize(16), transforms.CenterCrop(16),
                                          transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
        m.ENVIRONMENT = {'test': 'synthetic CPU'}
        m.ENGINE_HASH = m.code_fingerprint()
        m.validate_settings()
        for k in [1, 2]:
            name = f'holdout_ADM_k{k}_rep0'
            folder = m.MANIFEST_ROOT / name
            folder.mkdir()
            for split, counts in m.EXPECTED_COUNTS.items():
                records = []
                for label, count in counts.items():
                    for i in range(count):
                        generator = 'ADM' if label == 0 or split == 'test' else ['Midjourney', 'wukong'][i % k]
                        p = m.DATA_ROOT / generator / split / f'{name}_{label}_{i}.png'
                        p.parent.mkdir(parents=True, exist_ok=True)
                        rng = np.random.default_rng(int(m.digest_json([name, split, label, i])[:8], 16))
                        image = rng.integers(0 if label == 0 else 100, 155 if label == 0 else 255,
                                             size=(20, 24, 3), dtype=np.uint8)
                        Image.fromarray(image).save(p)
                        records.append({'path': '/content/drive/MyDrive/GenImage_Compact_shared/' + p.relative_to(m.DATA_ROOT).as_posix(),
                                        'label': label, 'source_generator': generator, 'sha256': m.file_digest(p)})
                pd.DataFrame(records).to_csv(folder / f'{split}.csv', index=False)
        cls.frames, cls.info = m.load_experiment('holdout_ADM_k2_rep0')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def make_config(self, mode):
        m = self.m
        config = {**self.info, 'seed': 17, 'mode': mode}
        config['protocol_hash'] = m.digest_json(config)
        return config

    def test_default_plan_and_syntax(self):
        self.assertEqual(self.default_plan_size, 24)
        ids = [c['id'] for c in self.nb['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for i, c in enumerate(self.nb['cells']):
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), f'cell-{i}', 'exec')

    def test_metrics_and_calibration_boundaries(self):
        m = self.m
        perfect = m.classification_metrics([0, 1, 0, 1], [0., 1., 0., 1.])
        self.assertEqual(perfect['auroc'], 1.)
        self.assertEqual(perfect['accuracy'], 1.)
        self.assertEqual(perfect['brier'], 0.)
        self.assertEqual(perfect['ece'], 0.)
        self.assertEqual(m.reliability_bins([0,1], [0.,1.])['count'].sum(), 2)
        self.assertEqual(m.classification_metrics([0,1], [.5,.5])['f1_fake'], 2/3)
        for bad in [[float('nan'), .2], [-.1, 1], [0, 1.1]]:
            with self.assertRaises(ValueError):
                m.classification_metrics([0,1], bad)

    def test_paths_hashes_and_leakage(self):
        m = self.m
        with self.assertRaises(ValueError):
            m.resolve_image_path('../escape.png')
        with self.assertRaises(ValueError):
            m.resolve_image_path('/tmp/elsewhere.png')
        csv = m.MANIFEST_ROOT / 'holdout_ADM_k2_rep0/val.csv'
        original = csv.read_bytes()
        try:
            val = pd.read_csv(csv)
            train = self.frames['train']
            val.loc[0, ['path','sha256']] = train.loc[0, ['path','sha256']].to_list()
            val.to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                m.load_experiment('holdout_ADM_k2_rep0')
        finally:
            csv.write_bytes(original)
        p = Path(self.frames['train'].iloc[0].resolved_path)
        before = p.read_bytes()
        try:
            p.write_bytes(before + b'changed')
            with self.assertRaisesRegex(ValueError, 'Content differs'):
                m.load_experiment('holdout_ADM_k2_rep0')
        finally:
            p.write_bytes(before)

    def test_cache_and_deterministic_degradations(self):
        m = self.m
        x, y = m.deterministic_features(self.frames['val'])
        calls = m.ENCODER.calls
        second, labels = m.deterministic_features(self.frames['val'])
        self.assertEqual(calls, m.ENCODER.calls)
        self.assertTrue(torch.equal(x, second))
        self.assertTrue(torch.equal(y, labels))
        row = self.frames['val'].iloc[0]
        for condition in ['clean', 'blur','jpeg','resize','noise','lowpass']:
            with Image.open(row.resolved_path) as im:
                transform = m.ImageTransform(m.PREPROCESS, condition=condition)
                first = transform(im.convert('RGB'), row.sha256)
                second = transform(im.convert('RGB'), row.sha256)
            self.assertTrue(torch.equal(first,second), condition)
            self.assertTrue(torch.isfinite(first).all())
        a = m.feature_identity(self.frames['val'], 'clean')
        b = m.feature_identity(self.frames['val'].iloc[::-1], 'clean')
        self.assertNotEqual(m.digest_json(a), m.digest_json(b))

    def test_training_resume_evaluation_and_exports(self):
        m = self.m
        for mode in ['baseline','augmented']:
            config = self.make_config(mode)
            full = self.root / f'full-{mode}'
            resumed = self.root / f'resumed-{mode}'
            full.mkdir(); resumed.mkdir()
            m.train_one(full, self.frames, config)
            original_save = m.atomic_torch
            def interrupt_after_epoch(path, value):
                original_save(path, value)
                if Path(path).name == 'last.pt' and value['state']['epoch'] == 1:
                    raise KeyboardInterrupt('synthetic interruption')
            m.atomic_torch = interrupt_after_epoch
            try:
                with self.assertRaises(KeyboardInterrupt):
                    m.train_one(resumed, self.frames, config)
            finally:
                m.atomic_torch = original_save
            m.train_one(resumed, self.frames, config)
            a = torch.load(full/'best_head.pt', weights_only=True)
            b = torch.load(resumed/'best_head.pt', weights_only=True)
            for key in a['head_state_dict']:
                self.assertTrue(torch.equal(a['head_state_dict'][key], b['head_state_dict'][key]), (mode,key))
            self.assertEqual(len(pd.read_csv(resumed/'training_history.csv')), 2)
            for condition in ['clean','noise']:
                m.evaluate_one(full, self.frames, config, condition)
                calls = m.ENCODER.calls
                m.evaluate_one(full, self.frames, config, condition)
                self.assertEqual(calls, m.ENCODER.calls)
                _, _, directory = m.evaluation_identity(full, config, condition)
                pred = pd.read_csv(directory/'test_predictions.csv')
                self.assertEqual(len(pred), 4)
                self.assertTrue({'image_id','label','prob_fake','source_generator','experiment','mode','seed'} <= set(pred))
                self.assertTrue((directory/'test_diagnostics.png').is_file())
                if condition == 'noise':
                    (directory/'test_predictions.csv').unlink()
                    with self.assertRaises(FileNotFoundError):
                        m.evaluate_one(full, self.frames, config, condition)

    def test_exclusive_lock(self):
        m = self.m
        d = self.root/'lock-test'
        d.mkdir()
        with m.run_lock(d):
            with self.assertRaises(RuntimeError):
                with m.run_lock(d):
                    pass
        self.assertFalse((d/'RUNNING.lock').exists())

    def test_plan_runner_resume_limit_and_summaries(self):
        m = self.m
        # Exercise the actual plan, runner, and aggregation cells on tiny images.
        m.OUTPUT_ROOT = self.root / 'plan-results'
        m.HOLDOUTS = ['ADM']; m.K_VALUES = [1, 2]; m.REPEATS = [0]
        m.TRAINING_MODES = ['baseline', 'augmented']
        m.EVAL_CONDITIONS = ['clean']
        m.MAX_RUNS_THIS_SESSION = 2
        m.package_versions = lambda: {'test': 'synthetic CPU'}
        code = [''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code']
        plan = next(c for c in code if c.startswith('validate_settings()'))
        runner = next(c for c in code if c.startswith('session_rows ='))
        summary = next(c for c in code if c.startswith('rows, missing ='))
        exec(compile(plan, 'plan-cell', 'exec'), m.__dict__)
        self.assertEqual(len(m.PLAN), 4)
        exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        self.assertEqual(len(list(m.OUTPUT_ROOT.glob('runs/*/TRAINING_COMPLETE.json'))), 2)
        exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        self.assertEqual(len(list(m.OUTPUT_ROOT.glob('runs/*/TRAINING_COMPLETE.json'))), 4)
        calls = m.ENCODER.calls
        exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        self.assertEqual(m.ENCODER.calls, calls)
        exec(compile(summary, 'summary-cell', 'exec'), m.__dict__)
        results = pd.read_csv(m.SUMMARY_DIR/'all_results.csv')
        paired = pd.read_csv(m.SUMMARY_DIR/'augmentation_deltas.csv')
        self.assertEqual(len(results), 4)
        self.assertEqual(len(paired), 2)
        self.assertTrue((m.SUMMARY_DIR/'baseline_clean_auroc_heatmap.png').is_file())
        # Changing training settings must reject existing run identities.
        m.LEARNING_RATE *= 2
        exec(compile(plan, 'plan-cell', 'exec'), m.__dict__)
        with self.assertRaisesRegex(ValueError, 'Protocol changed'):
            exec(compile(runner, 'runner-cell', 'exec'), m.__dict__)
        m.LEARNING_RATE /= 2


if __name__ == '__main__':
    unittest.main()
