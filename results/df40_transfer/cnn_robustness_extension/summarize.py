"""Summarize the matched k=7 CNN/ViT robustness results with audit status."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
VIT = ROOT.parent.parent / 'vit_robustness/results/per_run.csv'


def main():
    paths = [ROOT / 'export/adm/all_results.csv'] + sorted((ROOT / 'export/extension').glob('*/all_results.csv'))
    if len(paths) != 8:
        raise ValueError('Expected eight holdout summaries')
    cnn = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    vit = pd.read_csv(VIT)
    if len(cnn) != 144 or len(vit) != 144:
        raise ValueError('Expected 144 evaluations per model')
    cnn = cnn.rename(columns={'variant': 'mode'})
    cnn['condition'] = cnn['condition'].map({'clean': 'clean', 'jpeg60': 'jpeg', 'resize_112': 'resize'})
    keys = ['holdout', 'rep', 'mode', 'condition']
    paired = cnn.merge(vit, on=keys, suffixes=('_cnn', '_vit'), validate='one_to_one')
    if len(paired) != 144 or paired['condition'].isna().any():
        raise ValueError('Evaluation pair mismatch')
    audit = json.loads((ROOT / 'AUDIT_INVENTORY.json').read_text())['holdouts']
    strict = {name: all(r['status'] == 'passed' for r in rows) for name, rows in audit.items()}
    paired['cnn_strict_clean_replay_passed'] = paired['holdout'].map(strict)
    cols = keys + ['accuracy_cnn', 'accuracy_vit', 'auroc_cnn', 'auroc_vit',
                   'ece_cnn', 'ece_vit', 'cnn_strict_clean_replay_passed']
    paired[cols].to_csv(ROOT / 'paired_per_run.csv', index=False)
    metrics = ['accuracy_cnn', 'accuracy_vit', 'auroc_cnn', 'auroc_vit', 'ece_cnn', 'ece_vit']
    holdout = paired.groupby(['holdout', 'mode', 'condition'])[metrics].mean().reset_index()
    holdout['cnn_strict_clean_replay_passed'] = holdout['holdout'].map(strict)
    holdout.to_csv(ROOT / 'paired_by_holdout.csv', index=False)
    macro = holdout.groupby(['mode', 'condition'])[metrics].mean().reset_index()
    macro.to_csv(ROOT / 'paired_macro_provisional.csv', index=False)
    print(macro.to_string(index=False))
    print('Strict replay failed in:', [name for name, passed in strict.items() if not passed])


if __name__ == '__main__':
    main()
