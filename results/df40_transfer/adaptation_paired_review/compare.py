"""Describe matched objective differences without treating reused tests as independent."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'generator_objectives_seeds/per_holdout_seed.csv'


def main():
    data = pd.read_csv(SOURCE)
    base = data[data.objective == 'baseline'].set_index(['seed', 'holdout'])
    if len(base) != 24 or not base.index.is_unique:
        raise ValueError('Expected 24 unique baseline seed/holdout pairs')
    frames = []
    for objective in ('adversarial', 'contrastive'):
        other = data[data.objective == objective].set_index(['seed', 'holdout'])
        if len(other) != 24 or not other.index.is_unique or set(other.index) != set(base.index):
            raise ValueError(f'Unmatched pairs: {objective}')
        joined = other.join(base, lsuffix='_new', rsuffix='_base', validate='one_to_one')
        out = joined[['accuracy_new', 'accuracy_base', 'auroc_new', 'auroc_base',
                      'real_fp_new', 'real_fp_base', 'fake_tp_new', 'fake_tp_base']].copy()
        for metric in ('accuracy', 'auroc', 'real_fp', 'fake_tp'):
            out[f'delta_{metric}'] = out[f'{metric}_new'] - out[f'{metric}_base']
        out.insert(0, 'objective', objective)
        frames.append(out.reset_index())
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(ROOT / 'paired_deltas.csv', index=False, lineterminator='\n')
    for objective, group in result.groupby('objective'):
        delta = group.delta_accuracy
        print(objective, 'pairs', len(delta), 'better', int((delta > 0).sum()),
              'worse', int((delta < 0).sum()), 'tied', int((delta == 0).sum()),
              'mean', round(delta.mean(), 6), 'range', round(delta.min(), 6), round(delta.max(), 6))


if __name__ == '__main__':
    main()
