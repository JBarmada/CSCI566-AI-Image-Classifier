"""Export completed CNN benchmark prediction and provenance files."""
import tarfile
from pathlib import Path

ROOT = Path('/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/notebook')
OUT = Path('/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/cnn_prediction_export_v1.tar.gz')
FILES = ('test_predictions.csv', 'val_predictions.csv', 'config.json', 'TRAINING_COMPLETE.json')


def main():
    runs = sorted((ROOT / 'runs').glob('holdout_*'))
    if len(runs) != 192 or OUT.exists():
        raise ValueError('Expected 192 runs and a new output path')
    with tarfile.open(OUT, 'x:gz') as archive:
        for run in runs:
            for name in FILES:
                path = run / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                archive.add(path, arcname=f'runs/{run.name}/{name}')
    print(f'Exported {len(runs)} runs to {OUT}')


if __name__ == '__main__':
    main()
