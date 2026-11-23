"""Rebuild paired robustness summaries from independently audited V100 outputs."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'cnn_robustness_extension' / 'paired_per_run.csv'
NEW = ROOT / 'outputs'
KEYS = ['holdout', 'rep', 'mode', 'condition']
METRICS = ['accuracy', 'auroc', 'ece']
CONDITIONS = {'clean': 'clean', 'jpeg60': 'jpeg', 'resize_112': 'resize'}


def read_new():
    evidence = json.loads((ROOT / 'INDEPENDENT_AUDIT.json').read_text())
    if evidence['status'] != 'passed' or evidence['evaluations'] != 72:
        raise ValueError('V100 independent audit did not pass')
    audited = {(r['run'], r['condition']) for r in evidence['comparisons']}
    rows = []
    for path in sorted(NEW.glob('*/runs/*/*/result.json')):
        item = json.loads(path.read_text())
        if (item['run_name'], item['condition']) not in audited:
            raise ValueError(f'Unaudited result {path}')
        rows.append({'holdout': item['holdout'], 'rep': item['rep'],
                     'mode': item['variant'], 'condition': CONDITIONS[item['condition']],
                     **{f'{m}_new': item[m] for m in METRICS}})
    result = pd.DataFrame(rows)
    if len(result) != 72 or result.duplicated(KEYS).any():
        raise ValueError('Expected 72 uniquely keyed V100 results')
    return result


def main():
    old = pd.read_csv(OLD)
    if len(old) != 144 or old.duplicated(KEYS).any():
        raise ValueError('Expected 144 unique existing paired rows')
    joined = old.merge(read_new(), on=KEYS, how='left', validate='one_to_one', indicator=True)
    selected = joined['_merge'] == 'both'
    if selected.sum() != 72 or not joined.loc[~selected, 'cnn_strict_clean_replay_passed'].all():
        raise ValueError('Unexpected replacement set or pre-existing failed checks')
    for metric in METRICS:
        joined.loc[selected, f'{metric}_cnn'] = joined.loc[selected, f'{metric}_new']
    joined['cnn_strict_clean_replay_passed'] = True
    joined['cnn_gpu'] = selected.map({True: 'V100', False: 'P100'})
    cols = list(old.columns) + ['cnn_gpu']
    paired = joined[cols]
    paired.to_csv(ROOT / 'paired_per_run.csv', index=False, lineterminator='\n')
    metrics = [f'{m}_{model}' for m in METRICS for model in ('cnn', 'vit')]
    holdout = paired.groupby(['holdout', 'mode', 'condition'])[metrics].mean().reset_index()
    holdout.to_csv(ROOT / 'paired_by_holdout.csv', index=False, lineterminator='\n')
    macro = holdout.groupby(['mode', 'condition'])[metrics].mean().reset_index()
    macro.to_csv(ROOT / 'paired_macro.csv', index=False, lineterminator='\n')
    report = {'old_paired_sha256': hashlib.sha256(OLD.read_bytes()).hexdigest(),
              'rows': len(paired), 'v100_replacements': int(selected.sum()),
              'max_cnn_metric_change': {m: float((paired[f'{m}_cnn'] - old[f'{m}_cnn']).abs().max())
                                        for m in METRICS}}
    (ROOT / 'SUMMARY_VERIFICATION.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
