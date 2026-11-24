"""Contrast real false alarms with fake recall in two exploratory face sets."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def main():
    df = pd.read_csv(ROOT / 'results/df40_transfer/shared_results_v1/metrics.csv')
    still = pd.read_csv(ROOT / 'results/df40_transfer/still_face_v1/metrics.csv')
    examples = []
    for model in ('cnn', 'vit'):
        video = df[(df.model == model) & (df['mode'] == 'baseline') & (df.method == 'simswap')].iloc[0]
        photo = still[(still.model == model) & (still['mode'] == 'baseline') & (still.method == 'insight')].iloc[0]
        display = 'CNN' if model == 'cnn' else 'ViT'
        examples.extend([(display + ' • DF40 SimSwap', video.fp, video.real, video.tp, video.fake),
                         (display + ' • DeepFakeFace InsightFace', photo.false_positive, photo.real_n,
                          photo.true_positive, photo.fake_n)])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(examples))
    for offset, field, color, label in [(-.19, 1, '#a63b30', 'Real images falsely flagged'),
                                        (.19, 3, '#1769aa', 'Fake images detected')]:
        numerators = [int(item[field]) for item in examples]
        denominators = [int(item[field + 1]) for item in examples]
        values = [100 * n / d for n, d in zip(numerators, denominators)]
        ax.barh(y + offset, values, height=.32, color=color, label=label)
        for position, value, n, total in zip(y + offset, values, numerators, denominators):
            ax.text(value + 1, position, f'{n}/{total}', va='center', fontsize=10)
    ax.set_yticks(y, [item[0] for item in examples])
    ax.set(xlim=(0, 120), ylim=(len(examples)-.5, -.5), xlabel='Images flagged fake (%)',
           title='High fake detection can coincide with many false alarms')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.text(.5, .01, 'Baseline models, threshold 0.5. Sets and manipulation methods differ; causes are unknown.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'08_face_error_tradeoff.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
