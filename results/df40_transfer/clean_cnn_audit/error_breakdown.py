"""Summarize matched k=7 clean CNN and ViT errors by true class."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_matched import CNN, ROOT, VIT, read


def summarize(run):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    if config['k'] != 7:
        return []
    cnn = read(run / 'test_predictions.csv')
    paths = list((VIT / config['holdout'] / 'runs' / run.name / 'evaluations').glob('clean_*/test_predictions.csv'))
    if len(paths) != 1:
        raise ValueError(f'Expected one clean ViT prediction file: {run.name}')
    vit = {row['sha256']: row for row in read(paths[0])}
    if len(cnn) != 560 or len(vit) != 560 or {r['sha256'] for r in cnn} != set(vit):
        raise ValueError(f'Image mismatch: {run.name}')
    if any(row['label'] != vit[row['sha256']]['label'] for row in cnn):
        raise ValueError(f'Label mismatch: {run.name}')
    y = np.array([int(row['label']) for row in cnn])
    cp = np.array([float(row['probability_fake']) for row in cnn])
    vp = np.array([float(vit[row['sha256']]['prob_fake']) for row in cnn])
    if not (np.isfinite(cp).all() and np.isfinite(vp).all()):
        raise ValueError(f'Invalid probabilities: {run.name}')
    rows = []
    for label in (0, 1):
        mask = y == label
        if mask.sum() != 280:
            raise ValueError(f'Unexpected class count: {run.name}')
        cwrong = (cp[mask] >= .5) != label
        vwrong = (vp[mask] >= .5) != label
        rows.append({'run': run.name, 'holdout': config['holdout'], 'repeat': config['rep'],
                     'mode': config['mode'], 'true_class': 'real' if label == 0 else 'fake',
                     'n': int(mask.sum()), 'cnn_error': int(cwrong.sum()),
                     'vit_error': int(vwrong.sum()), 'both_error': int((cwrong & vwrong).sum()),
                     'cnn_only_correct': int((~cwrong & vwrong).sum()),
                     'vit_only_correct': int((cwrong & ~vwrong).sum()),
                     'cnn_confident_error': int((cwrong & ((cp[mask] <= .1) | (cp[mask] >= .9))).sum()),
                     'vit_confident_error': int((vwrong & ((vp[mask] <= .1) | (vp[mask] >= .9))).sum())})
    return rows


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = [row for run in sorted(CNN.glob('holdout_*')) for row in summarize(run)]
    if len(rows) != 96:
        raise ValueError(f'Expected 96 class-specific rows, got {len(rows)}')
    write(ROOT / 'error_breakdown_per_run.csv', rows)
    fields = ['cnn_error', 'vit_error', 'both_error', 'cnn_only_correct', 'vit_only_correct',
              'cnn_confident_error', 'vit_confident_error']
    summary = []
    for mode in ('baseline', 'augmented'):
        for label in ('real', 'fake'):
            for holdout in sorted({r['holdout'] for r in rows}):
                group = [r for r in rows if r['mode'] == mode and r['true_class'] == label and r['holdout'] == holdout]
                if len(group) != 3:
                    raise ValueError('Expected three repeats per holdout')
                summary.append({'mode': mode, 'true_class': label, 'holdout': holdout, 'n_per_run': 280,
                                **{f'{field}_mean_count': float(np.mean([r[field] for r in group])) for field in fields}})
    write(ROOT / 'error_breakdown_by_holdout.csv', summary)
    print('Verified 48 matched k=7 runs; saved error counts by class and holdout.')


if __name__ == '__main__':
    main()
