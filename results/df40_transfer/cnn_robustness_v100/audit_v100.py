"""Audit and compare V100 and previously recorded P100 robustness outputs."""
import csv
import json
import math
from pathlib import Path
import audit

HOME = Path.home() / 'cnn_robustness_v100_20261123'
OLD = audit.BASE / 'results/cnn_robustness_extension_v1/outputs'
HOLDOUTS = {3: 'stable_diffusion_v_1_4', 4: 'stable_diffusion_v_1_5',
            5: 'VQDM', 6: 'wukong'}


def read_scores(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return {r['sha256']: r for r in rows}


def compare(new, old):
    left, right = read_scores(new), read_scores(old)
    if len(left) != 560 or left.keys() != right.keys():
        raise ValueError('P100/V100 prediction identities mismatch')
    if any(left[k]['label'] != right[k]['label'] for k in left):
        raise ValueError('P100/V100 labels mismatch')
    differences = [abs(float(left[k]['probability_fake']) -
                       float(right[k]['probability_fake'])) for k in left]
    changed = sum((float(left[k]['probability_fake']) >= .5) !=
                  (float(right[k]['probability_fake']) >= .5) for k in left)
    return {'rows': len(left), 'max_abs_probability_difference': max(differences),
            'changed_labels': changed}


def check_extra_metrics(folder):
    report = json.loads((folder / 'result.json').read_text())
    rows = read_scores(folder / 'test_predictions.csv').values()
    pairs = [(int(r['label']), float(r['probability_fake'])) for r in rows]
    bins = [[] for _ in range(10)]
    for label, score in pairs:
        bins[min(int(score * 10), 9)].append((label, score))
    ece = sum(abs(sum(p for _, p in b) - sum(y for y, _ in b))
              for b in bins if b) / len(pairs)
    brier = sum((p - y) ** 2 for y, p in pairs) / len(pairs)
    clipped = [(y, min(1 - 1e-7, max(1e-7, p))) for y, p in pairs]
    logloss = -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                   for y, p in clipped) / len(pairs)
    if any(abs(report[k] - v) > 1e-9 for k, v in
           (('ece', ece), ('brier', brier), ('log_loss', logloss))):
        raise ValueError(f'Calibration metric mismatch: {folder}')


def main():
    checks = []
    for index, name in HOLDOUTS.items():
        folder = HOME / f'task_14492182_{index}' / 'outputs' / name
        runs = sorted((folder / 'runs').iterdir())
        if len(runs) != 6:
            raise ValueError(f'Expected six runs for {name}')
        for run in runs:
            passed = audit.audit_run(folder, run)
            for condition in ('clean', 'jpeg60', 'resize_112'):
                check_extra_metrics(run / condition)
                old = OLD / name / 'runs' / run.name / condition / 'test_predictions.csv'
                new = run / condition / 'test_predictions.csv'
                checks.append({'holdout': name, 'run': run.name, 'condition': condition,
                               'clean_max_difference': passed['clean_max_probability_difference'],
                               **compare(new, old)})
    if len(checks) != 72:
        raise ValueError('Expected 72 condition/run comparisons')
    output = {'status': 'passed', 'clean_tolerance': .001,
              'runs': 24, 'evaluations': len(checks),
              'metrics_checked': ['accuracy', 'auroc', 'ece', 'brier', 'log_loss'],
              'comparisons': checks}
    (HOME / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': output['status'], 'evaluations': len(checks),
                      'max_clean_difference': max(r['clean_max_difference'] for r in checks),
                      'changed_labels': sum(r['changed_labels'] for r in checks)}))


if __name__ == '__main__':
    main()
