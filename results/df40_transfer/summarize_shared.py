"""Summarize frozen-model predictions by manipulation."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'shared_results_v1'


def metrics(rows):
    real = [float(r['prob_fake']) for r in rows if r['label'] == '0']
    fake = [float(r['prob_fake']) for r in rows if r['label'] == '1']
    if not real or not fake:
        raise ValueError('Both classes required')
    tp = sum(p >= 0.5 for p in fake)
    fp = sum(p >= 0.5 for p in real)
    tn, fn = len(real) - fp, len(fake) - tp
    auc = sum((f > r) + 0.5 * (f == r) for f in fake for r in real) / (len(real) * len(fake))
    return {'real': len(real), 'fake': len(fake), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'accuracy': (tp + tn) / len(rows), 'fake_recall': tp / len(fake),
            'real_false_positive_rate': fp / len(real), 'auroc': auc,
            'f1': (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0}


def main():
    with (ROOT / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    results = []
    for architecture in ('cnn', 'vit'):
        for mode in ('baseline', 'augmented'):
            selected = [r for r in rows if r['model'] == architecture and r['mode'] == mode]
            if len(selected) != 500:
                raise ValueError('Incomplete model predictions')
            for method in ('simswap', 'fomm', 'wav2lip'):
                comparison = [r for r in selected if r['method'] in ('real', method)]
                results.append({'model': architecture, 'mode': mode, 'method': method,
                                **metrics(comparison)})
    (ROOT / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    with (ROOT / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
