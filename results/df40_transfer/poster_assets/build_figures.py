"""Build poster-ready figures from the checked project result tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COLORS = {'CNN': '#2166ac', 'ViT': '#b35806'}


def save(fig, name):
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def diversity():
    cnn = pd.read_csv(ROOT / 'results/df40_transfer/clean_cnn_audit/equal_weight_holdout_summary.csv')
    vit = pd.read_csv(ROOT / 'results/vit_benchmark/summary/macro_across_holdouts.csv')
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label, data, mode, value in [('CNN', cnn, 'variant', 'mean_holdout_accuracy'),
                                      ('ViT', vit, 'mode', 'accuracy_mean')]:
        for variant, linestyle in [('baseline', '-'), ('augmented', '--')]:
            rows = data[data[mode] == variant].sort_values('k')
            assert rows['k'].tolist() == [1, 2, 4, 7]
            ax.plot(rows['k'], 100 * rows[value], color=COLORS[label], linestyle=linestyle,
                    marker='o', linewidth=2.2, label=f'{label} {variant}')
    ax.set(title='Higher clean accuracy with more training generators',
           xlabel='Number of training generators (k)', ylabel='Accuracy (%)', ylim=(50, 100))
    ax.set_xticks([1, 2, 4, 7])
    ax.grid(axis='y', alpha=.2)
    ax.legend(ncol=2, fontsize=9, handlelength=4.5, columnspacing=1.4)
    save(fig, '01_generator_diversity')


def robustness():
    data = pd.read_csv(ROOT / 'results/df40_transfer/cnn_robustness_v100/paired_macro.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, mode in zip(axes, ['baseline', 'augmented']):
        rows = data[data['mode'] == mode].set_index('condition').loc[['clean', 'jpeg', 'resize']]
        x = np.arange(3)
        for offset, model in [(-.18, 'CNN'), (.18, 'ViT')]:
            ax.bar(x + offset, 100 * rows[f'accuracy_{model.lower()}'], .36,
                   color=COLORS[model], label=model)
        ax.set_xticks(x, ['Clean', 'JPEG 60', 'Resize'])
        ax.set(title=mode.title(), ylim=(0, 100), ylabel='Accuracy (%)')
        ax.grid(axis='y', alpha=.2)
    axes[1].legend()
    fig.suptitle('Augmentation improves resistance to image degradation')
    save(fig, '02_robustness')


def transfer():
    df = pd.read_csv(ROOT / 'results/df40_transfer/shared_results_v1/metrics.csv')
    sf = pd.read_csv(ROOT / 'results/df40_transfer/still_face_v1/metrics.csv')
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(2)
    for offset, model, label in [(-.18, 'cnn', 'CNN'), (.18, 'vit', 'ViT')]:
        a = df[(df.model == model) & (df['mode'] == 'baseline') & (df.method == 'fomm')].iloc[0]
        b = sf[(sf.model == model) & (sf['mode'] == 'baseline') & (sf.method == 'insight')].iloc[0]
        values = [100 * a.real_false_positive_rate, 100 * b.real_fpr]
        bars = ax.bar(x + offset, values, .36, color=COLORS[label], label=label)
        ax.bar_label(bars, labels=[f'{int(a.fp)}/100', f'{int(b.false_positive)}/50'], padding=3)
    ax.set_xticks(x, ['DF40 real video crops', 'DeepFakeFace real photos'])
    ax.set(title='Real-image errors vary across these datasets', ylabel='Real images flagged fake (%)', ylim=(0, 110))
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    save(fig, '03_real_image_errors')


def main():
    diversity()
    robustness()
    transfer()
    print('Saved three figures in PNG and SVG formats')


if __name__ == '__main__':
    main()
