"""Verify objective outputs and aggregate one fixed run per holdout."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def audit(holdout, directory):
    result = json.loads((directory / 'RESULTS.json').read_text())
    replay = json.loads((directory / 'REPLAY.json').read_text())
    if result['status'] != 'passed' or replay['status'] != 'passed' or holdout in result['domains']:
        raise ValueError(f'Provenance or replay failed for {holdout}')
    rows = []
    membership = None
    for item in result['results']:
        name = item['objective']
        predictions = read(directory / (name + '_predictions.csv'))
        current = {r['sha256']: r['label'] for r in predictions}
        if membership is not None and current != membership:
            raise ValueError(f'Objective test membership differs: {holdout}')
        membership = current
        labels = np.array([int(r['label']) for r in predictions])
        scores = np.array([float(r['prob_fake']) for r in predictions])
        if len(predictions) != 560 or len(set(r['sha256'] for r in predictions)) != 560:
            raise ValueError(f'Unexpected predictions: {holdout}/{name}')
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Invalid scores')
        metrics = {'accuracy': float(np.mean((scores >= .5) == labels)),
                   'auroc': float(roc_auc_score(labels, scores)),
                   'real_fp': int(np.sum((scores >= .5) & (labels == 0))),
                   'fake_tp': int(np.sum((scores >= .5) & (labels == 1)))}
        if any(abs(metrics[k] - item[k]) > 1e-10 for k in metrics):
            raise ValueError(f'Metric mismatch: {holdout}/{name}')
        checkpoint = directory / (name + '_head.pt')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != replay['checks'][name]['checkpoint_sha256']:
            raise ValueError(f'Checkpoint mismatch: {holdout}/{name}')
        selection = json.loads((directory / (name + '_selection.json')).read_text())
        rows.append({'holdout': holdout, 'objective': name, **metrics,
                     'selected_epoch': selection['epoch'], 'validation_auc': selection['validation_auc']})
    if len(rows) != 3 or {r['objective'] for r in rows} != {'baseline', 'adversarial', 'contrastive'}:
        raise ValueError('Expected all three objectives')
    return rows


def main():
    sources = [('ADM', ROOT.parent / 'generator_objectives/results_v1')]
    sources += [(p.name.removeprefix('results_14483084_'), p) for p in sorted(ROOT.glob('results_14483084_*')) if p.is_dir()]
    if len(sources) != 8:
        raise ValueError('Expected eight holdouts')
    rows = [row for holdout, directory in sources for row in audit(holdout, directory)]
    with (ROOT / 'per_holdout_objectives.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ('baseline', 'adversarial', 'contrastive'):
        group = [r for r in rows if r['objective'] == name]
        print(name, 'mean accuracy', round(np.mean([r['accuracy'] for r in group]), 6),
              'mean AUROC', round(np.mean([r['auroc'] for r in group]), 6),
              'selected epoch 20', sum(r['selected_epoch'] == 20 for r in group))
    for name in ('adversarial', 'contrastive'):
        pairs = [(next(r for r in rows if r['holdout'] == h and r['objective'] == name),
                  next(r for r in rows if r['holdout'] == h and r['objective'] == 'baseline')) for h, _ in sources]
        print(name, 'accuracy wins/ties/losses', *(sum(np.sign(a['accuracy'] - b['accuracy']) == n for a, b in pairs) for n in (1, 0, -1)))
    print('Audited 24 result files and checkpoint replays on eight holdouts.')


if __name__ == '__main__':
    main()
