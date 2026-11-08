"""Plot the verified clean benchmark tables; error bars describe holdout variation."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {'baseline': '#2563ab', 'augmented': '#cc5b29'}


def trends(root, output):
    frame = pd.read_csv(root / 'summary/macro_across_holdouts.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), layout='constrained')
    for ax, metric, label in zip(axes, ['accuracy', 'auroc', 'brier'],
                               ['Accuracy at threshold 0.5', 'AUROC', 'Brier score (lower is better)']):
        for mode, group in frame.groupby('mode'):
            ax.errorbar(group.k, group[metric + '_mean'], yerr=group[metric + '_std'],
                        marker='o', capsize=4, label=mode.title(), color=COLORS[mode])
        ax.set(xlabel='Training generators (k)', ylabel=label, xticks=[1, 2, 4, 7])
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Frozen CLIP ViT-L/14: clean unseen-generator tests\n'
                 'Equal-weight mean of 8 holdouts; bars = SD of holdout means, not confidence intervals',
                 fontsize=12)
    fig.savefig(output / 'performance_vs_k.png', dpi=190)
    plt.close(fig)


def heatmaps(root, output):
    frame = pd.read_csv(root / 'summary/by_holdout_k.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3), layout='constrained')
    for ax, mode in zip(axes, ['baseline', 'augmented']):
        table = frame[frame['mode'] == mode].pivot(index='holdout', columns='k', values='auroc_mean')
        labels = [s.replace('stable_diffusion_v_', 'SD ') for s in table.index]
        mesh = ax.imshow(table, vmin=.65, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set(xticks=range(4), xticklabels=[1, 2, 4, 7], yticks=range(len(table)),
               yticklabels=labels, xlabel='Training generators (k)', title=mode.title())
        for i in range(len(table)):
            for j in range(4):
                value = table.iloc[i, j]
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color='white' if value > .9 else '#122333')
    fig.colorbar(mesh, ax=axes, label='Mean test AUROC', shrink=.85)
    fig.suptitle('Clean test AUROC by held-out generator\nMean of three dataset repeats; one classifier seed')
    fig.savefig(output / 'holdout_auroc.png', dpi=190)
    plt.close(fig)


def reliability(root, output):
    frame = pd.read_csv(root / 'metadata/calibration_bins.csv.gz')
    frame = frame[(frame.split == 'test') & (frame.k == 7)].copy()
    fig, ax = plt.subplots(figsize=(6.2, 5.1), layout='constrained')
    for mode, group in frame.groupby('mode'):
        group['sum_probability'] = group['count'] * group.mean_prob_fake.fillna(0)
        group['sum_fake'] = group['count'] * group.observed_fake_fraction.fillna(0)
        bins = group.groupby('bin')[['count', 'sum_probability', 'sum_fake']].sum()
        bins = bins[bins['count'] > 0]
        ax.plot(bins.sum_probability / bins['count'], bins.sum_fake / bins['count'],
                'o-', color=COLORS[mode], label=mode.title())
    ax.plot([0, 1], [0, 1], '--', color='#777777', label='Perfect reliability')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability of fake',
           ylabel='Observed fake fraction', title='Raw probability reliability at k=7\n'
           'Descriptive pooling of fixed test images reused across repeats')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(output / 'raw_reliability_k7.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    destination = args.root / 'figures'
    destination.mkdir(exist_ok=True)
    trends(args.root, destination)
    heatmaps(args.root, destination)
    reliability(args.root, destination)
