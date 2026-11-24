"""Plot paired augmentation effects from the audited k=7 comparison."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'cnn_robustness_v100' / 'paired_by_holdout.csv'


def main():
    data = pd.read_csv(DATA)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True, sharey=True)
    for row, model in enumerate(('vit', 'cnn')):
        for col, condition in enumerate(('clean', 'jpeg', 'resize')):
            ax = axes[row, col]
            selected = data[data.condition == condition].set_index(['holdout', 'mode'])
            base = selected.xs('baseline', level='mode')
            augmented = selected.xs('augmented', level='mode')
            paired = (augmented.select_dtypes('number') - base.select_dtypes('number')).sort_index()
            x = 100 * paired[f'accuracy_{model}']
            y = 100 * paired[f'auroc_{model}']
            ax.axhline(0, color='#999999', linewidth=.8)
            ax.axvline(0, color='#999999', linewidth=.8)
            ax.scatter(x, y, color='#2166ac', s=32, alpha=.8)
            ax.scatter([x.mean()], [y.mean()], color='#b2182b', marker='D', s=65, zorder=3)
            ax.set_title(f'{model.upper()} — {condition.upper()}')
            ax.grid(alpha=.15)
            if row == 1:
                ax.set_xlabel('Accuracy change (percentage points)')
            if col == 0:
                ax.set_ylabel('AUROC change (points × 100)')
    fig.suptitle('Augmentation changes threshold accuracy and ranking differently')
    fig.text(.5, .02, 'Dots: held-out generators; diamond: equal-weight average. Positive = augmented higher.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(ROOT / f'04_accuracy_auroc_tradeoffs.{suffix}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
