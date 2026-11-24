"""Make a compact poster candidate from audited ViT JPEG macro metrics."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    table = pd.read_csv(ROOT.parent / 'cnn_robustness_v100/paired_macro.csv')
    rows = table[table.condition == 'jpeg'].set_index('mode').loc[['baseline', 'augmented']]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    fields = [('accuracy_vit', 'Accuracy at threshold 0.5', '#1769aa'),
              ('auroc_vit', 'AUROC × 100', '#a13d24')]
    for y, (key, label, color) in enumerate(fields):
        baseline, augmented = 100 * rows[key].to_numpy()
        delta = augmented - baseline
        ax.barh(y, delta, height=.42, color=color)
        ax.text(max(delta, 0) + .15, y, f'{delta:+.2f} points', va='center',
                ha='left', fontsize=12, weight='bold')
        ax.text(.15, y + .32, f'{baseline:.2f} → {augmented:.2f}', va='center', fontsize=10)
    ax.axvline(0, color='#333333', linewidth=.9)
    ax.set_yticks([0, 1], [item[1] for item in fields])
    ax.set(xlim=(-1.3, 9.2), ylim=(1.6, -.6), xlabel='Change in percentage points (augmented minus baseline)',
           title='JPEG: accuracy improved; AUROC changed very little')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .01, 'Values: baseline → augmented. Equal-weight mean across eight held-out generators.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / f'05_vit_jpeg_accuracy_auroc.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
