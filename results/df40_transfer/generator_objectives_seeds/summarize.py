"""Verify and summarize fixed seed replications of objective comparisons."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'generator_objectives_extended'
spec = importlib.util.spec_from_file_location('objective_audit', PREVIOUS / 'summarize.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
HOLDOUTS = ('ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
            'stable_diffusion_v_1_5', 'VQDM', 'wukong')


def source(holdout, seed):
    if seed == 17 and holdout == 'ADM':
        return ROOT.parent / 'generator_objectives/results_v1'
    if seed == 17:
        return PREVIOUS / f'results_14483084_{holdout}'
    return ROOT / f'results_14483127_{holdout}_seed{seed}'


def main():
    rows = []
    for holdout in HOLDOUTS:
        manifest_hashes = None
        for seed in (17, 29, 43):
            directory = source(holdout, seed)
            record = json.loads((directory / 'RESULTS.json').read_text())
            current_hashes = {k: v['manifest_sha256'] for k, v in record['cache'].items()}
            if manifest_hashes is not None and current_hashes != manifest_hashes:
                raise ValueError(f'Manifest changed between seeds: {holdout}')
            manifest_hashes = current_hashes
            if seed != 17 and record.get('optimization_seed') != seed:
                raise ValueError(f'Wrong seed: {holdout}/{seed}')
            rows.extend({'seed': seed, **r} for r in audit_module.audit(holdout, directory))
    if len(rows) != 72:
        raise ValueError('Expected 72 objective results')
    with (ROOT / 'per_holdout_seed.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for holdout in HOLDOUTS:
        for objective in ('baseline', 'adversarial', 'contrastive'):
            group = [r for r in rows if r['holdout'] == holdout and r['objective'] == objective]
            aggregates.append({'holdout': holdout, 'objective': objective,
                               'mean_accuracy': float(np.mean([r['accuracy'] for r in group])),
                               'mean_auroc': float(np.mean([r['auroc'] for r in group])),
                               'mean_real_fp': float(np.mean([r['real_fp'] for r in group])),
                               'mean_fake_tp': float(np.mean([r['fake_tp'] for r in group]))})
    with (ROOT / 'mean_by_holdout.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    for objective in ('baseline', 'adversarial', 'contrastive'):
        group = [r for r in aggregates if r['objective'] == objective]
        print(objective, 'accuracy', round(np.mean([r['mean_accuracy'] for r in group]), 6),
              'auroc', round(np.mean([r['mean_auroc'] for r in group]), 6))
        if objective != 'baseline':
            base = {r['holdout']: r for r in aggregates if r['objective'] == 'baseline'}
            print('holdout accuracy wins/ties/losses', *(sum(np.sign(r['mean_accuracy'] - base[r['holdout']]['mean_accuracy']) == x for r in group) for x in (1, 0, -1)))
    print('Verified 72 predictions/heads; averaged three seeds within eight holdouts.')


if __name__ == '__main__':
    main()
