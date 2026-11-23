"""Plot equal-weight matched clean-benchmark confidence metrics."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(ROOT / 'matched_confidence_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for mode, style in [('baseline', '-'), ('augmented', '--')]:
        subset = data[data['mode'] == mode].sort_values('k')
        for model, color in [('cnn', '#235a9f'), ('vit', '#b14f21')]:
            for ax, metric, label in [(axes[0], 'brier', 'Brier score'),
                                      (axes[1], 'ece10', 'Expected calibration error')]:
                ax.plot(subset['k'], subset[model + '_' + metric], marker='o', linestyle=style,
                        color=color, label=f'{model.upper()} {mode}')
                ax.set_ylabel(label + ' (lower is better)')
                ax.set_xlabel('Training generators (k)')
                ax.set_xticks([1, 2, 4, 7])
                ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Clean benchmark: mean across 8 holdouts and 3 dataset repeats')
    fig.tight_layout()
    fig.savefig(ROOT / 'matched_confidence.png', dpi=180)


if __name__ == '__main__':
    main()
