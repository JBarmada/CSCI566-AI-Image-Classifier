"""Summarize fixed-prompt scores by manipulation against shared real images."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize(rows, method):
    real = [float(r['prob_fake_prompt']) for r in rows if r['method'] == 'real']
    fake = [float(r['prob_fake_prompt']) for r in rows if r['method'] == method]
    fp = sum(p >= .5 for p in real)
    tp = sum(p >= .5 for p in fake)
    return {'method': method, 'real': len(real), 'fake': len(fake),
            'fp': fp, 'tn': len(real) - fp, 'tp': tp, 'fn': len(fake) - tp,
            'real_fpr': fp / len(real), 'fake_recall': tp / len(fake),
            'auroc': sum((f > r) + .5 * (f == r) for f in fake for r in real) / (len(real) * len(fake))}


def main():
    with (ROOT / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 500 or len({r['sha256'] for r in rows}) != 500:
        raise ValueError('Incomplete or duplicate predictions')
    result = [summarize(rows, m) for m in ('simswap', 'fomm', 'wav2lip')]
    (ROOT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
