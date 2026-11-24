"""Collect the 48 existing ViT JPEG prediction tables without rerunning models."""
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path('/project2/yzhao010_1531/csci566-genimage/results/vit_robustness_k7_v1')


def main():
    files = sorted(ROOT.glob('*/runs/*/evaluations/jpeg_*/test_predictions.csv'))
    if len(files) != 48:
        raise ValueError(f'Expected 48 ViT JPEG prediction files; got {len(files)}')
    with ZipFile(Path(sys.argv[1]), 'x', compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'Archived {len(files)} previously saved JPEG prediction tables')


if __name__ == '__main__':
    main()
