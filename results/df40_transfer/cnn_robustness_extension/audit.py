"""Independently audit a completed CNN robustness holdout."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = Path('/project2/yzhao010_1531/csci566-genimage')
RUNS = BASE / 'results/cnn_benchmark/notebook/runs'
OUT = BASE / 'results/cnn_robustness_extension_v1/outputs'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def audit_run(folder, run):
    source = RUNS / run.name
    training = json.loads((source / 'TRAINING_COMPLETE.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    if digest(source / 'best.pt') != training['checkpoint_sha256'] or config['protocol_hash'] != training['protocol_hash']:
        raise ValueError(f'Checkpoint identity mismatch: {run.name}')
    reference = {r['sha256']: r for r in read(source / 'test_predictions.csv')}
    manifest = {r['sha256']: r for r in read(source / 'input_test.csv')}
    if len(reference) != 560 or set(reference) != set(manifest):
        raise ValueError('Source prediction and manifest membership differ')
    differences = []
    for condition in ('clean', 'jpeg60', 'resize_112'):
        item = run / condition
        report = json.loads((item / 'result.json').read_text())
        settings = json.loads((item / 'evaluation_config.json').read_text())
        marker = json.loads((item / 'COMPLETE.json').read_text())
        if (marker['evaluation_hash'] != report['evaluation_hash'] or
                settings['evaluation_hash'] != marker['evaluation_hash'] or
                report['checkpoint_sha256'] != training['checkpoint_sha256'] or
                report['test_manifest_sha256'] != digest(source / 'input_test.csv')):
            raise ValueError('Evaluation identity mismatch')
        predictions = read(item / 'test_predictions.csv')
        ids = [r['sha256'] for r in predictions]
        if len(ids) != 560 or len(set(ids)) != 560 or set(ids) != set(reference):
            raise ValueError('Prediction membership mismatch')
        if any(r['label'] != reference[r['sha256']]['label'] for r in predictions):
            raise ValueError('Prediction labels mismatch')
        y = np.array([int(r['label']) for r in predictions])
        p = np.array([float(r['probability_fake']) for r in predictions])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probability')
        if abs(accuracy_score(y, p >= .5) - report['accuracy']) > 1e-9 or abs(roc_auc_score(y, p) - report['auroc']) > 1e-9:
            raise ValueError('Reported metrics mismatch')
        if condition == 'clean':
            differences = [abs(float(r['probability_fake']) - float(reference[r['sha256']]['probability_fake']))
                           for r in predictions]
    largest = max(differences)
    if largest > .001:
        raise ValueError(f'Clean replay differs by {largest}')
    return {'run': run.name, 'clean_max_probability_difference': largest}


def main():
    folder = OUT / sys.argv[1]
    runs = sorted((folder / 'runs').iterdir())
    if len(runs) != 6:
        raise ValueError('Expected six k=7 models')
    audit = [audit_run(folder, run) for run in runs]
    result = {'status': 'passed', 'runs': len(runs), 'evaluations': len(runs) * 3,
              'clean_tolerance': .001, 'details': audit}
    (folder / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
