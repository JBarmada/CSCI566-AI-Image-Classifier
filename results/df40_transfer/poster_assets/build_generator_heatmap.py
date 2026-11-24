"""Make a labeled heatmap of baseline ViT accuracy by held-out generator."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def main():
    data = pd.read_csv(ROOT / 'results/vit_benchmark/summary/by_holdout_k.csv')
    table = data[data['mode'] == 'baseline'].pivot(index='holdout', columns='k', values='accuracy_mean')
    table = table.loc[:, [1, 2, 4, 7]] * 100
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    cells = ax.imshow(table, cmap='Blues', vmin=50, vmax=100, aspect='auto')
    for row in range(len(table)):
        for col in range(len(table.columns)):
            value = table.iloc[row, col]
            ax.text(col, row, f'{value:.0f}', ha='center', va='center',
                    color='white' if value >= 85 else 'black', fontsize=10)
    ax.set_xticks(range(4), ['1', '2', '4', '7'])
    labels = [name.replace('stable_diffusion_v_1_', 'Stable Diffusion 1.').replace('glide', 'GLIDE')
              .replace('wukong', 'WuKong') for name in table.index]
    ax.set_yticks(range(len(table)), labels)
    ax.set(xlabel='Training generator families (k)', ylabel='Held-out generator',
           title='Clean ViT accuracy varies by held-out generator')
    fig.colorbar(cells, ax=ax, label='Accuracy (%)', shrink=.8)
    fig.text(.5, .01, 'Baseline ViT; mean across three data repeats per cell; fixed image budget.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'07_vit_generator_heatmap.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
