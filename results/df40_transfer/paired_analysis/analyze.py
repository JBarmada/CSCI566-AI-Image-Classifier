"""Pair saved CNN and ViT predictions and summarize outcomes by class."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'shared_results_v1/predictions.csv'


def load_pairs():
    with SOURCE.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    index = {}
    for row in rows:
        key = (row['mode'], row['sha256'], row['model'])
        if key in index:
            raise ValueError(f'Duplicate model and image: {key}')
        index[key] = row
    pairs = []
    for mode in ('baseline', 'augmented'):
        for model in ('cnn', 'vit'):
            if sum(k[0] == mode and k[2] == model for k in index) != 500:
                raise ValueError('Expected 500 predictions per model')
        for key, cnn in index.items():
            if key[0] != mode or key[2] != 'cnn':
                continue
            vit = index[(mode, key[1], 'vit')]
            for field in ('label', 'method', 'video_group', 'image_path'):
                if cnn[field] != vit[field]:
                    raise ValueError(f'Metadata mismatch: {field}')
            label = int(cnn['label'])
            c, v = float(cnn['prob_fake']) >= .5, float(vit['prob_fake']) >= .5
            pairs.append({'mode': mode, 'method': cnn['method'], 'label': label,
                          'sha256': key[1], 'image_path': cnn['image_path'],
                          'video_group': cnn['video_group'], 'cnn_fake': int(c),
                          'vit_fake': int(v), 'cnn_correct': int(c == label),
                          'vit_correct': int(v == label)})
    if len(pairs) != 1000:
        raise ValueError('Wrong paired count')
    return pairs


def summarize(pairs):
    summary = []
    for mode in ('baseline', 'augmented'):
        for method in ('real', 'simswap', 'fomm', 'wav2lip'):
            rows = [r for r in pairs if r['mode'] == mode and r['method'] == method]
            counts = Counter((r['cnn_correct'], r['vit_correct']) for r in rows)
            expected = 200 if method == 'simswap' else 100
            if len(rows) != expected:
                raise ValueError('Unexpected sample count')
            summary.append({'mode': mode, 'method': method, 'n': len(rows),
                            'both_correct': counts[(1, 1)], 'cnn_only_correct': counts[(1, 0)],
                            'vit_only_correct': counts[(0, 1)], 'both_wrong': counts[(0, 0)],
                            'disagree': counts[(1, 0)] + counts[(0, 1)]})
    return summary


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    pairs = load_pairs()
    summary = summarize(pairs)
    write(ROOT / 'pairs.csv', pairs)
    write(ROOT / 'summary.csv', summary)
    print('\n'.join(str(row) for row in summary))


if __name__ == '__main__':
    main()
