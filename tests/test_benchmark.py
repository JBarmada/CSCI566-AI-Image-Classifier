"""Regression checks for dataset mistakes that previously affected the study."""
from collections import defaultdict
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/benchmark'))
import audit_benchmark as audit
from build_benchmark import choose, unique
from image_inventory import properties, write_csv
from near_duplicates import candidates


class DatasetTests(unittest.TestCase):
    def fixture(self, root, suffix='rep0'):
        name = 'holdout_ADM_k1_' + suffix
        experiment = {'experiment': name, 'holdout': 'ADM', 'k': 1, 'train_generators': ['BigGAN']}
        inventory, splits = {}, {}
        for split in ['train', 'val', 'test']:
            rows = []
            for label in ['0', '1']:
                for i in range(2):
                    path = f'{split}/{label}/{i}.jpg'
                    sha = hashlib.sha256(path.encode()).hexdigest()
                    row = {'path': path, 'label': label,
                           'source_generator': 'ADM' if split == 'test' or label == '0' else 'BigGAN',
                           'source_id': sha, 'sha256': sha, 'pixel_sha256': sha}
                    rows.append(row)
                    inventory[path] = dict(row, role=split)
            splits[split] = rows
            write_csv(root / 'manifests' / name / f'{split}.csv', rows)
        return experiment, inventory, splits

    def check(self, root, experiment, inventory, real=None, test=None):
        with patch.object(audit, 'COUNTS', {'train': 2, 'val': 2, 'test': 2}):
            audit.check_experiment(root, experiment, inventory,
                                   {} if real is None else real, {} if test is None else test,
                                   defaultdict(set))

    def test_missing_rows_and_holdout_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp, inventory, splits = self.fixture(root)
            self.check(root, exp, inventory)
            file = root / 'manifests' / exp['experiment'] / 'train.csv'
            write_csv(file, splits['train'][:-1])
            with self.assertRaisesRegex(ValueError, 'counts'):
                self.check(root, exp, inventory)
            splits['train'][-1]['source_generator'] = 'ADM'
            write_csv(file, splits['train'])
            with self.assertRaisesRegex(ValueError, 'wrong generators'):
                self.check(root, exp, inventory)

    def test_changed_test_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp, inventory, splits = self.fixture(root)
            tests, real = {}, {}
            self.check(root, exp, inventory, real, tests)
            row = splits['test'][-1]
            row.update(path='new.jpg', sha256='new', source_id='new', pixel_sha256='new')
            inventory['new.jpg'] = dict(row, role='test')
            write_csv(root / 'manifests' / exp['experiment'] / 'test.csv', splits['test'])
            with self.assertRaisesRegex(ValueError, 'test membership changed'):
                self.check(root, exp, inventory, real, tests)

    def test_cross_split_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp, inventory, splits = self.fixture(root)
            splits['val'][0] = splits['train'][0]
            write_csv(root / 'manifests' / exp['experiment'] / 'val.csv', splits['val'])
            with self.assertRaisesRegex(ValueError, 'split leakage'):
                self.check(root, exp, inventory)

    def test_actual_jpeg_settings_are_checked(self):
        for quality, sampling, expected in [(96, 0, 1), (95, 0, 0), (96, 2, 0)]:
            buffer = io.BytesIO()
            Image.new('RGB', (32, 32), 'red').save(buffer, format='JPEG', quality=quality, subsampling=sampling)
            self.assertEqual(properties(buffer.getvalue())['jpeg_match'], expected)

    def test_duplicate_pixels_and_shortfalls_are_not_silently_accepted(self):
        rows = [{'path': str(i), 'source_id': str(i), 'sha256': str(i), 'pixel_sha256': 'same'} for i in range(2)]
        self.assertEqual(len(unique(rows)), 1)
        with self.assertRaises(ValueError):
            choose(rows, 3, 'too-small')

    def test_perceptual_candidate_search_matches_brute_force(self):
        import random
        values = [random.Random(i).getrandbits(64) for i in range(100)]
        values += [values[0] ^ 15, values[1] ^ (1 << 63)]
        rows = [{'path': str(i), 'phash': f'{v:016x}', 'pixel_sha256': str(i)} for i, v in enumerate(values)]
        found = {(int(p['left']), int(p['right'])) for p in candidates(rows)}
        expected = {(j, i) for i, v in enumerate(values) for j in range(i)
                    if (v ^ values[j]).bit_count() <= 4}
        self.assertEqual(found, expected)


if __name__ == '__main__':
    unittest.main()
