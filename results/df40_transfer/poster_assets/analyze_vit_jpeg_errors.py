"""Audit class-specific errors in saved, paired ViT JPEG predictions."""
import io
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import pandas as pd

OUT = Path(__file__).resolve().parent
PATTERN = re.compile(r'([^/]+)/runs/holdout_.+_k7_rep(\d+)__(baseline|augmented)__seed\d+/')


def load(archive):
    result = {}
    for name in archive.namelist():
        match = PATTERN.match(name)
        if not match:
            raise ValueError(f'Unexpected file: {name}')
        holdout, rep, mode = match.groups()
        frame = pd.read_csv(io.BytesIO(archive.read(name)))
        if frame.image_id.duplicated().any() or set(frame.label.unique()) != {0, 1}:
            raise ValueError(f'Invalid prediction labels or identities: {name}')
        result[(holdout, int(rep), mode)] = frame.set_index('image_id')
    if len(result) != 48:
        raise ValueError('Expected 48 paired files')
    return result


def compare(data):
    output = []
    for holdout, rep in sorted({key[:2] for key in data}):
        baseline = data[(holdout, rep, 'baseline')].sort_index()
        augmented = data[(holdout, rep, 'augmented')].sort_index()
        if not baseline.index.equals(augmented.index) or not baseline.label.equals(augmented.label):
            raise ValueError(f'Unmatched predictions: {holdout} repeat {rep}')
        for label, name in ((0, 'real'), (1, 'fake')):
            b = baseline[baseline.label == label].prob_fake
            a = augmented[augmented.label == label].prob_fake
            output.append({'holdout': holdout, 'rep': rep, 'class': name, 'n': len(b),
                           'baseline_flagged_fake': int((b >= .5).sum()),
                           'augmented_flagged_fake': int((a >= .5).sum()),
                           'baseline_rate': float((b >= .5).mean()),
                           'augmented_rate': float((a >= .5).mean()),
                           'baseline_mean_score': float(b.mean()), 'augmented_mean_score': float(a.mean()),
                           'score_lower_with_augmentation': int((a < b).sum()),
                           'flagged_to_unflagged': int(((b >= .5) & (a < .5)).sum()),
                           'unflagged_to_flagged': int(((b < .5) & (a >= .5)).sum())})
    return pd.DataFrame(output)


def main():
    with ZipFile(Path(sys.argv[1])) as archive:
        pairs = compare(load(archive))
    pairs.to_csv(OUT / 'vit_jpeg_paired_class_rates.csv', index=False)
    holdouts = pairs.groupby(['holdout', 'class'])[['baseline_rate', 'augmented_rate']].mean().reset_index()
    macro = holdouts.groupby('class')[['baseline_rate', 'augmented_rate']].mean()
    macro['change_percentage_points'] = 100 * (macro.augmented_rate - macro.baseline_rate)
    macro.to_csv(OUT / 'vit_jpeg_class_macro.csv')
    print(macro.to_string())


if __name__ == '__main__':
    main()
