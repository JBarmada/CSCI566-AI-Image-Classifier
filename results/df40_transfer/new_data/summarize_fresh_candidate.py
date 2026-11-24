"""Summarize frozen candidate predictions using the predeclared thresholds."""
import csv
import json
from pathlib import Path
import sys

THRESHOLDS = {('cnn', 'baseline'): 0.9990234375000001,
              ('cnn', 'augmented'): 0.7763671875000001,
              ('vit', 'baseline'): 0.615590274333954,
              ('vit', 'augmented'): 0.5530025362968446,
              ('vit', 'face_adapted'): 0.5676835775375367}
METHODS = ('simswap', 'fomm', 'wav2lip', 'inswap', 'pirender')


def auc(real, fake):
    wins = sum((f > r) + .5 * (f == r) for f in fake for r in real)
    return wins / (len(real) * len(fake))


def summarize(rows, model, mode, method, threshold, name):
    subset = [r for r in rows if r['model'] == model and r['mode'] == mode]
    real = [float(r['prob_fake']) for r in subset if r['method'] == 'real']
    fake = [float(r['prob_fake']) for r in subset if r['method'] == method]
    if len(real) != 58 or len(fake) != 29:
        raise ValueError('Incomplete predictions')
    fp = sum(score >= threshold for score in real)
    tp = sum(score >= threshold for score in fake)
    return {'model': model, 'mode': mode, 'method': method, 'threshold_name': name,
            'threshold': threshold, 'real_fp': fp, 'real_n': len(real), 'fake_tp': tp,
            'fake_n': len(fake), 'real_fpr': fp / len(real), 'fake_recall': tp / len(fake),
            'balanced_accuracy': (1 - fp / len(real) + tp / len(fake)) / 2,
            'auroc': auc(real, fake)}


def main():
    folder = Path(sys.argv[1])
    acceptance = json.loads((folder / 'ACCEPTANCE.json').read_text())
    if acceptance.get('status') != 'passed' or acceptance.get('scores') != 1015:
        raise ValueError('Inference checks did not pass')
    with (folder / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    result = [summarize(rows, model, mode, method, threshold, name)
              for (model, mode), selected in THRESHOLDS.items() for method in METHODS
              for name, threshold in (('original_0.5', .5), ('previous_validation', selected))]
    with (folder / 'metrics.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f'Saved {len(result)} exploratory metric rows')


if __name__ == '__main__':
    main()
