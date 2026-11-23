"""Summarize pooled k=7 reliability bins for the matched clean runs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_matched import CNN, ROOT, VIT, read


def main():
    bins = {(model, mode, i): [] for model in ('cnn', 'vit')
            for mode in ('baseline', 'augmented') for i in range(10)}
    paths = sorted(CNN.glob('holdout_*_k7_rep*__*__seed17'))
    if len(paths) != 48:
        raise ValueError('Expected 48 k=7 runs')
    for run in paths:
        cnn = read(run / 'test_predictions.csv')
        holdout, mode = cnn[0]['holdout'], cnn[0]['mode']
        files = list((VIT / holdout / 'runs' / run.name / 'evaluations').glob('clean_*/test_predictions.csv'))
        if len(files) != 1:
            raise ValueError(f'Missing ViT test: {run.name}')
        for model, rows, score in [('cnn', cnn, 'probability_fake'), ('vit', read(files[0]), 'prob_fake')]:
            for row in rows:
                p = float(row[score])
                bins[(model, mode, min(int(p * 10), 9))].append((p, int(row['label'])))
    output = []
    for (model, mode, index), values in bins.items():
        if values:
            output.append({'model': model, 'mode': mode, 'bin': index,
                           'count': len(values), 'mean_score': np.mean([p for p, y in values]),
                           'fake_fraction': np.mean([y for p, y in values])})
    with (ROOT / 'reliability_k7.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, mode in zip(axes, ('baseline', 'augmented')):
        ax.plot([0, 1], [0, 1], 'k--', alpha=.5, label='Ideal')
        for model in ('cnn', 'vit'):
            selected = [r for r in output if r['model'] == model and r['mode'] == mode]
            ax.plot([r['mean_score'] for r in selected], [r['fake_fraction'] for r in selected],
                    marker='o', label=model.upper())
        ax.set(title=mode.title(), xlabel='Mean predicted fake score', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed fake fraction')
    axes[1].legend()
    fig.suptitle('Clean k=7 reliability, pooled across holdouts and repeats')
    fig.tight_layout()
    fig.savefig(ROOT / 'reliability_k7.png', dpi=180)
    print('Wrote pooled reliability bins and figure')


if __name__ == '__main__':
    main()
