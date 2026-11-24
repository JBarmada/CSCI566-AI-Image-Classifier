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
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    series = [('accuracy_vit', '#1769aa', 'o', '-', 'Accuracy at 0.5'),
              ('auroc_vit', '#a13d24', 's', '--', 'AUROC × 100')]
    for key, color, marker, style, label in series:
        values = 100 * rows[key].to_numpy()
        ax.plot([0, 1], values, color=color, marker=marker, linestyle=style,
                linewidth=2.5, markersize=8, label=label)
        for x, value in enumerate(values):
            ax.annotate(f'{value:.1f}', (x, value), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=11)
    ax.set_xticks([0, 1], ['Baseline', 'Augmented'])
    ax.set(xlim=(-.25, 1.25), ylim=(50, 100), ylabel='Score (percent or AUROC × 100)',
           title='JPEG: ViT accuracy rises while mean AUROC stays similar')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='lower right', frameon=False)
    fig.text(.5, .01, 'Equal-weight average across eight held-out generators; JPEG quality 60.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / f'05_vit_jpeg_accuracy_auroc.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
