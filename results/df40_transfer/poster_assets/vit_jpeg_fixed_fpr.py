"""Describe test-set ROC performance at a fixed empirical false-positive rate."""
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
from analyze_vit_jpeg_errors import load

OUT = Path(__file__).resolve().parent
FPR_TARGET = .10


def main():
    with ZipFile(Path(sys.argv[1])) as archive:
        data = load(archive)
    rows = []
    for (holdout, rep, mode), frame in sorted(data.items()):
        counts = frame.groupby('prob_fake').label.agg(['sum', 'count']).sort_index(ascending=False)
        fake = counts['sum'].cumsum()
        real = (counts['count'] - counts['sum']).cumsum()
        real_n = int((frame.label == 0).sum())
        fake_n = int((frame.label == 1).sum())
        eligible = fake[real <= FPR_TARGET * real_n]
        recall = float(eligible.max() / fake_n) if len(eligible) else 0.0
        rows.append({'holdout': holdout, 'rep': rep, 'mode': mode,
                     'empirical_fpr_target': FPR_TARGET, 'test_roc_recall': recall})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / 'vit_jpeg_fixed_fpr_per_run.csv', index=False)
    holdouts = per_run.groupby(['holdout', 'mode']).test_roc_recall.mean().reset_index()
    summary = holdouts.groupby('mode').test_roc_recall.mean()
    summary.to_csv(OUT / 'vit_jpeg_fixed_fpr_macro.csv')
    print(summary.to_string())


if __name__ == '__main__':
    main()
