"""Plot provisional matched k=7 robustness accuracy and AUROC."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(ROOT / 'paired_macro_provisional.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    positions = np.arange(3)
    for ax, mode in zip(axes, ('baseline', 'augmented')):
        rows = data[data['mode'] == mode].set_index('condition').loc[['clean', 'jpeg', 'resize']]
        ax.bar(positions - .18, rows['accuracy_cnn'], width=.36, label='CNN', color='#235a9f')
        ax.bar(positions + .18, rows['accuracy_vit'], width=.36, label='ViT', color='#b14f21')
        ax.set_xticks(positions, ['Clean', 'JPEG 60', 'Resize 112'])
        ax.set(title=mode.title(), ylim=(0, 1), ylabel='Accuracy')
        ax.grid(axis='y', alpha=.2)
    axes[1].legend()
    fig.suptitle('Provisional k=7 robustness: 8 holdouts, equal weight')
    fig.tight_layout()
    fig.savefig(ROOT / 'provisional_accuracy.png', dpi=180)


if __name__ == '__main__':
    main()
