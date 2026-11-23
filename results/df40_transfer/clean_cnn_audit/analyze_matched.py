"""Verify and analyze matched clean CNN/ViT test predictions."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
VIT = ROOT.parent.parent / 'audit_sources/vit_benchmark'
CNN = ROOT / 'prediction_export/runs'


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def metrics(labels, probability):
    predicted = probability >= .5
    accurate = predicted == labels
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = sum(np.mean(bins == i) * abs(np.mean(probability[bins == i]) - np.mean(labels[bins == i]))
              for i in range(10) if np.any(bins == i))
    return {'accuracy': float(np.mean(accurate)), 'auroc': float(roc_auc_score(labels, probability)),
            'brier': float(np.mean((probability - labels) ** 2)), 'ece10': float(ece),
            'confident_errors': int(np.sum((~accurate) & ((probability >= .9) | (probability <= .1))))}


def analyze(run):
    cnn = read(run / 'test_predictions.csv')
    config = json.loads((run / 'config.json').read_text())
    marker = json.loads((run / 'TRAINING_COMPLETE.json').read_text())
    if config['protocol_hash'] != marker['protocol_hash']:
        raise ValueError(f'CNN provenance mismatch: {run.name}')
    path = VIT / config['holdout'] / 'runs' / run.name / 'evaluations'
    files = list(path.glob('clean_*/test_predictions.csv'))
    if len(files) != 1:
        raise ValueError(f'Expected one ViT clean test: {run.name}')
    vit = read(files[0])
    by_hash = {r['sha256']: r for r in vit}
    if len(cnn) != 560 or len(by_hash) != 560 or len({r['sha256'] for r in cnn}) != 560:
        raise ValueError(f'Unexpected image counts: {run.name}')
    if {r['sha256'] for r in cnn} != set(by_hash):
        raise ValueError(f'Images differ: {run.name}')
    if any(r['label'] != by_hash[r['sha256']]['label'] for r in cnn):
        raise ValueError(f'Labels differ: {run.name}')
    y = np.array([int(r['label']) for r in cnn])
    c = np.array([float(r['probability_fake']) for r in cnn])
    v = np.array([float(by_hash[r['sha256']]['prob_fake']) for r in cnn])
    if not np.isfinite(c).all() or not np.isfinite(v).all() or np.any((c < 0) | (c > 1) | (v < 0) | (v > 1)):
        raise ValueError('Invalid probabilities')
    cm, vm = metrics(y, c), metrics(y, v)
    return {'run': run.name, 'holdout': config['holdout'], 'k': config['k'],
            'repeat': config['rep'], 'mode': config['mode'],
            **{'cnn_' + k: value for k, value in cm.items()},
            **{'vit_' + k: value for k, value in vm.items()},
            'both_wrong': int(np.sum(((c >= .5) != y) & ((v >= .5) != y))),
            'cnn_only_correct': int(np.sum(((c >= .5) == y) & ((v >= .5) != y))),
            'vit_only_correct': int(np.sum(((c >= .5) != y) & ((v >= .5) == y)))}


def main():
    runs = sorted(CNN.glob('holdout_*'))
    if len(runs) != 192:
        raise ValueError('Expected 192 matched runs')
    output = [analyze(run) for run in runs]
    with (ROOT / 'matched_confidence_per_run.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print('Verified 192 matched runs, 107520 paired image predictions')


if __name__ == '__main__':
    main()
