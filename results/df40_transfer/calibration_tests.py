"""Synthetic checks for leakage barriers and calibration."""
import unittest
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from calibration_evaluate import evaluate_frozen, fit_thresholds, identities, select_threshold


def sample(start, real_score, fake_score):
    rows = []
    for offset, method in enumerate(('real', 'simswap', 'fomm', 'wav2lip')):
        rows.append({'sha256': format(start + offset, '064x'), 'video_group': f'id{start + offset}_0001',
                     'method': method, 'label': '0' if method == 'real' else '1',
                     'model': 'vit', 'mode': 'baseline',
                     'prob_fake': str(real_score if method == 'real' else fake_score)})
    return rows


class CalibrationTests(unittest.TestCase):
    def test_tied_real_scores_and_strict_threshold(self):
        threshold = select_threshold([.2] * 9 + [.8], .1)
        self.assertGreater(threshold, .2)
        self.assertEqual(sum(p >= threshold for p in [.2] * 9 + [.8]), 1)

    def test_fresh_test_cannot_change_fitted_threshold(self):
        val, test, old = sample(1, .4, .7), sample(100, .3, .8), sample(200, .5, .5)
        frozen = fit_thresholds(val, old, .1)
        first = evaluate_frozen(frozen, test, old)['models'][0]
        test[0]['prob_fake'] = '.99'
        second = evaluate_frozen(frozen, test, old)['models'][0]
        self.assertEqual(first['threshold'], second['threshold'])
        self.assertNotEqual(first['fresh_test']['fomm']['fp'], second['fresh_test']['fomm']['fp'])

    def test_old_hash_and_known_identity_are_blocked(self):
        val, test, old = sample(1, .4, .7), sample(100, .3, .8), sample(200, .5, .5)
        test[0]['sha256'] = old[0]['sha256']
        with self.assertRaisesRegex(ValueError, 'hashes'):
            evaluate_frozen(fit_thresholds(val, old, .1), test, old)
        test[0]['sha256'] = format(100, '064x')
        test[0]['video_group'] = 'id200_other'
        with self.assertRaisesRegex(ValueError, 'identities'):
            evaluate_frozen(fit_thresholds(val, old, .1), test, old)

    def test_fit_refuses_previous_test_overlap(self):
        val, old = sample(1, .4, .7), sample(200, .5, .5)
        val[0]['video_group'] = old[0]['video_group']
        with self.assertRaisesRegex(ValueError, 'groups'):
            fit_thresholds(val, old, .1)

    def test_id_tokens_are_separated(self):
        self.assertEqual(identities('id3_id10_0001'), {'id3', 'id10'})
        self.assertEqual(identities('00251'), set())

    def test_cli_freezes_before_test_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, data in [('val', sample(1, .4, .7)), ('old', sample(200, .5, .5))]:
                with (root / (name + '.csv')).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=data[0])
                    writer.writeheader()
                    writer.writerows(data)
            script = Path(__file__).with_name('calibration_evaluate.py')
            frozen = root / 'frozen.json'
            old = root / 'old.csv'
            subprocess.run([sys.executable, str(script), 'fit', '--validation', str(root / 'val.csv'),
                            '--previous-test', str(old), '--output', str(frozen)], check=True)
            self.assertIn('validation_members', json.loads(frozen.read_text()))
            self.assertFalse((root / 'test.csv').exists())


if __name__ == '__main__':
    unittest.main()
