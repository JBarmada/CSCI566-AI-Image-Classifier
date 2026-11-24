"""Visualize class-specific changes in the ViT JPEG error audit."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(ROOT / 'vit_jpeg_class_macro.csv').set_index('class')
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    series = [('real', 'Real images flagged fake', '#17816b'),
              ('fake', 'Fake images detected', '#a63b30')]
    for y, (key, label, color) in enumerate(series):
        row = data.loc[key]
        delta = row.change_percentage_points
        ax.barh(y, delta, height=.40, color=color)
        ax.text(delta + 1, y, f'{delta:.1f} points', ha='left', va='center',
                color='white', weight='bold', fontsize=12)
        ax.text(.6, y, f'{100*row.baseline_rate:.1f}% → {100*row.augmented_rate:.1f}%',
                ha='left', va='center', fontsize=11)
    ax.axvline(0, color='#333333', linewidth=.9)
    ax.set_yticks([0, 1], [entry[1] for entry in series])
    ax.set(xlim=(-48, 23), ylim=(1.6, -.6),
           xlabel='Change in percentage points (augmented minus baseline)',
           title='JPEG: augmentation reduces false alarms and fake detections')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .01, 'ViT, threshold 0.5; baseline → augmented; eight holdouts weighted equally.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / f'06_vit_jpeg_tradeoff.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
