"""Compare audited CNN and ViT class-specific JPEG changes."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def rates():
    cnn = pd.read_csv(ROOT / 'cnn_jpeg_class_macro.csv').pivot(
        index='class', columns='mode', values='rate')
    vit = pd.read_csv(ROOT / 'vit_jpeg_class_macro.csv').set_index('class')
    return {'CNN': cnn.rename(columns={'baseline': 'baseline_rate',
                                       'augmented': 'augmented_rate'}), 'ViT': vit}


def main():
    data = rates()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for ax, (model, table) in zip(axes, data.items()):
        for y, (name, color) in enumerate([('real', '#17816b'), ('fake', '#a63b30')]):
            row = table.loc[name]
            change = 100 * (row.augmented_rate - row.baseline_rate)
            ax.barh(y, change, color=color, height=.42)
            ax.text(change + 1, y, f'{change:.1f}', color='white', fontsize=11,
                    weight='bold', va='center')
            ax.text(.7, y, f'{100*row.baseline_rate:.1f}% → {100*row.augmented_rate:.1f}%',
                    fontsize=10, va='center')
        ax.set(xlim=(-55, 26), ylim=(1.6, -.6), title=model,
               xlabel='Change in percentage points')
        ax.axvline(0, color='#333333', linewidth=.9)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks([0, 1], ['Real images flagged fake', 'Fake images detected'])
    fig.suptitle('JPEG: both detectors make fewer false alarms but miss more fakes', fontsize=15)
    fig.text(.5, .01, 'Threshold 0.5; baseline → augmented; eight holdouts weighted equally.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .92))
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / f'10_cnn_vit_jpeg_tradeoff.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
