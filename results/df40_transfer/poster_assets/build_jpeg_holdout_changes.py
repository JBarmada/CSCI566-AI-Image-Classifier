"""Plot fake-detection changes for each ViT JPEG holdout."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def display_name(name):
    return (name.replace('stable_diffusion_v_1_', 'Stable Diffusion 1.')
            .replace('glide', 'GLIDE').replace('wukong', 'WuKong'))


def main():
    data = pd.read_csv(ROOT / 'vit_jpeg_paired_class_rates.csv')
    fake = data[data['class'] == 'fake']
    means = fake.groupby('holdout')[['baseline_rate', 'augmented_rate']].mean()
    means['change'] = 100 * (means.augmented_rate - means.baseline_rate)
    means = means.sort_values('change')
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for y, (name, row) in enumerate(means.iterrows()):
        ax.barh(y, row['change'], color='#a63b30', height=.62)
        ax.text(row['change'] + .6, y, f"{row['change']:.1f}",
                color='white', weight='bold', va='center', fontsize=10)
        ax.text(.8, y, f"{100 * row.baseline_rate:.1f}% → {100 * row.augmented_rate:.1f}%",
                va='center', fontsize=10)
    ax.axvline(0, color='#333333', linewidth=.9)
    ax.set_yticks(range(len(means)), [display_name(name) for name in means.index])
    ax.set(xlim=(-50, 26), ylim=(len(means) - .4, -.6),
           xlabel='Change in fake detection (percentage points)',
           title='JPEG: fake detection falls for every held-out generator')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .01, 'ViT, threshold 0.5; baseline → augmented; mean of three dataset repeats.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(ROOT / f'09_jpeg_holdout_changes.{extension}', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
