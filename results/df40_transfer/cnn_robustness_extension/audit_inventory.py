"""Inventory all robustness checks while preserving strict replay failures."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import audit

BASE = Path('/project2/yzhao010_1531/csci566-genimage')
EXT = BASE / 'results/cnn_robustness_extension_v1/outputs'
ADM = BASE / 'results/cnn_robustness_k7_v1'
HOLDOUTS = ('ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
            'stable_diffusion_v_1_5', 'VQDM', 'wukong')


def clean_difference(run):
    source = audit.RUNS / run.name
    saved = {r['sha256']: r for r in audit.read(source / 'test_predictions.csv')}
    replay = audit.read(run / 'clean/test_predictions.csv')
    y = np.array([int(r['label']) for r in replay])
    new = np.array([float(r['probability_fake']) for r in replay])
    old = np.array([float(saved[r['sha256']]['probability_fake']) for r in replay])
    return {'max_probability_difference': float(np.max(np.abs(new - old))),
            'probabilities_over_001': int(np.sum(np.abs(new - old) > .001)),
            'changed_threshold_labels': int(np.sum((new >= .5) != (old >= .5))),
            'accuracy_delta': float(accuracy_score(y, new >= .5) - accuracy_score(y, old >= .5)),
            'auroc_delta': float(roc_auc_score(y, new) - roc_auc_score(y, old))}


def main():
    result = {}
    for name in HOLDOUTS:
        folder = ADM if name == 'ADM' else EXT / name
        runs = sorted((folder / 'runs').iterdir())
        if len(runs) != 6:
            raise ValueError(f'Expected six runs for {name}')
        records = []
        for run in runs:
            try:
                audit.audit_run(folder, run)
                status = 'passed'
            except ValueError as error:
                if not str(error).startswith('Clean replay differs by '):
                    raise
                status = 'strict_clean_replay_failed'
            records.append({'run': run.name, 'status': status, **clean_difference(run)})
        result[name] = records
    output = {'strict_tolerance': .001, 'holdouts': result,
              'status': 'complete_with_strict_replay_failures' if any(
                  r['status'] != 'passed' for rows in result.values() for r in rows) else 'passed'}
    path = EXT.parent / 'AUDIT_INVENTORY.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: sum(x['status'] != 'passed' for x in v) for k, v in result.items()}))


if __name__ == '__main__':
    main()
