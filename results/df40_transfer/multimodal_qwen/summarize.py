"""Compare multimodal labels and CNN/ViT predictions on the frozen subset."""
import csv
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent
RUNS = ROOT.parent / 'clean_cnn_audit/prediction_export/runs'
VIT = WORK / 'audit_sources/vit_benchmark/ADM/runs'


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def metrics(rows):
    label = [int(r['label']) for r in rows]
    pred = [int(r['prediction']) for r in rows]
    return {'n': len(rows), 'tn': sum(y == 0 and p == 0 for y, p in zip(label, pred)),
            'fp': sum(y == 0 and p == 1 for y, p in zip(label, pred)),
            'fn': sum(y == 1 and p == 0 for y, p in zip(label, pred)),
            'tp': sum(y == 1 and p == 1 for y, p in zip(label, pred)),
            'accuracy': sum(y == p for y, p in zip(label, pred)) / len(label)}


def compare(model, mode, manifest):
    run = f'holdout_ADM_k7_rep0__{mode}__seed17'
    if model == 'cnn':
        rows = read(RUNS / run / 'test_predictions.csv')
        score = 'probability_fake'
    else:
        files = list((VIT / run / 'evaluations').glob('clean_*/test_predictions.csv'))
        if len(files) != 1:
            raise ValueError('Expected one ViT clean result')
        rows = read(files[0])
        score = 'prob_fake'
    lookup = {r['sha256']: r for r in rows}
    result = []
    for item in manifest:
        match = lookup[item['sha256']]
        if item['label'] != match['label']:
            raise ValueError('Label mismatch')
        value = float(match[score])
        result.append({**item, 'prediction': int(value >= .5), 'score': value})
    return result


def main():
    manifest = read(ROOT / 'manifest.csv')
    raw = read(ROOT / 'results_v1/predictions.csv')
    if {r['sha256'] for r in raw} != {r['sha256'] for r in manifest}:
        raise ValueError('Multimodal image membership mismatch')
    valid = [r for r in raw if r['prediction'] in ('0', '1')]
    output = [{'model': 'Qwen2.5-VL-3B', 'mode': 'zero-shot',
               'valid': len(valid), 'invalid': len(raw) - len(valid), **metrics(valid)}]
    for model in ('cnn', 'vit'):
        for mode in ('baseline', 'augmented'):
            rows = compare(model, mode, manifest)
            item = {'model': model, 'mode': mode, 'valid': len(rows), 'invalid': 0, **metrics(rows)}
            item['auroc'] = roc_auc_score([int(r['label']) for r in rows], [r['score'] for r in rows])
            output.append(item)
    (ROOT / 'results_v1/comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
