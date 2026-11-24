"""Recompute CNN JPEG class rates using the audited P100/V100 selection."""
import io
from pathlib import Path
import tarfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
FILES = [BASE / 'cnn_robustness_extension/robustness_results_v1.tar.gz',
         BASE / 'cnn_robustness_v100/v100_results_v2.tar.gz']


def read_archive(path):
    output = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.name.endswith('/jpeg60/test_predictions.csv'):
                continue
            frame = pd.read_csv(io.BytesIO(archive.extractfile(member).read()))
            keys = frame[['holdout', 'rep', 'variant']].drop_duplicates()
            if len(keys) != 1 or frame.image_id.duplicated().any():
                raise ValueError(f'Invalid run membership: {member.name}')
            key = tuple(keys.iloc[0])
            if key in output or set(frame.label) != {0, 1}:
                raise ValueError(f'Duplicate run or invalid classes: {member.name}')
            output[key] = frame.set_index('image_id').sort_index()
    return output


def main():
    old, new = (read_archive(path) for path in FILES)
    if len(old) != 48 or len(new) != 24 or not set(new).issubset(old):
        raise ValueError('Unexpected P100/V100 run membership')
    source = {key: ('V100' if key in new else 'P100') for key in old}
    old.update(new)
    audit = pd.read_csv(BASE / 'cnn_robustness_v100/paired_per_run.csv')
    audit = audit[audit.condition == 'jpeg'].set_index(['holdout', 'rep', 'mode'])
    records = []
    for (holdout, rep, mode), frame in sorted(old.items()):
        other = old[(holdout, rep, 'augmented' if mode == 'baseline' else 'baseline')]
        if not frame.index.equals(other.index) or not frame.label.equals(other.label):
            raise ValueError(f'Unmatched model predictions for {holdout}, {rep}')
        row = audit.loc[(holdout, rep, mode)]
        accuracy = ((frame.probability_fake >= .5) == frame.label).mean()
        if abs(accuracy - row.accuracy_cnn) > 1e-9 or row.cnn_gpu != source[(holdout, rep, mode)]:
            raise ValueError(f'Audited accuracy or GPU mismatch for {holdout}, {rep}, {mode}')
        for label, name in ((0, 'real'), (1, 'fake')):
            selected = frame[frame.label == label]
            records.append({'holdout': holdout, 'rep': rep, 'mode': mode, 'class': name,
                            'n': len(selected), 'flagged_fake': int((selected.probability_fake >= .5).sum()),
                            'rate': float((selected.probability_fake >= .5).mean()), 'gpu': row.cnn_gpu})
    result = pd.DataFrame(records)
    result.to_csv(ROOT / 'cnn_jpeg_class_rates.csv', index=False)
    macro = result.groupby(['holdout', 'mode', 'class']).rate.mean().groupby(['mode', 'class']).mean()
    macro.to_csv(ROOT / 'cnn_jpeg_class_macro.csv')
    print(macro.to_string())


if __name__ == '__main__':
    main()
