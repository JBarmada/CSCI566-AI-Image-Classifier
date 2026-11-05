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
from unittest.mock import patch
import ast

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

NOTEBOOK = Path(__file__).resolve().parents[1] / 'ViT_Master_Experiments.ipynb'


def expanded_notebook():
    nb = json.loads(NOTEBOOK.read_text())
    engine = NOTEBOOK.parent / 'code/vit/engine'
    stages = {2: 'settings', 4: 'imports', 6: 'utilities', 8: 'manifests',
              10: 'features', 12: 'metrics', 14: 'training', 16: 'preflight',
              18: 'experiments', 20: 'summary'}
    for index, name in stages.items():
        text = (engine / f'{name}.py').read_text()
        if name == 'preflight':
            text = text[text.index('validate_settings()'):]
        if name == 'training':
            text = (engine / 'training_helpers.py').read_text() + '\n' + text
        nb['cells'][index]['source'] = text.splitlines(keepends=True)
    return nb


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode_image(self, images):
        self.calls += 1
        return torch.cat([images.mean((2, 3)), images.std((2, 3))], dim=1)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.nb = expanded_notebook()
        engine = NOTEBOOK.parent / 'code/vit/engine'
        source = ['import sys, types\nsys.modules.setdefault("clip", types.ModuleType("clip"))']
        for name in ['settings', 'imports', 'utilities', 'manifests', 'features',
                     'metrics', 'training_helpers', 'training']:
            text = (engine / (name + '.py')).read_text()
            if name == 'imports':
                source.append('REQUIRE_GPU = False')
                text = text.replace('from IPython.display import display', 'display = lambda value: None')
            source.append(text)
        source.append("DATASET_FINGERPRINT = 'synthetic'\nreport_progress = lambda *a, **k: None")
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
        cls.create_manifests()
        cls.frames, cls.info = m.load_experiment('holdout_ADM_k2_rep0')

    @classmethod
    def create_manifests(cls):
        m = cls.m
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


    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def make_config(self, mode):
        m = self.m
        config = {**self.info, 'seed': 17, 'mode': mode}
        config['protocol_hash'] = m.digest_json(config)
        return config
