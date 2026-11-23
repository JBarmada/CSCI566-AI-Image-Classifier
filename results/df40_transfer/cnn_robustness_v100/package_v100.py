"""Package audited V100 outputs without source images or model weights."""
import tarfile
from pathlib import Path

ROOT = Path.home() / 'cnn_robustness_v100_20261123'
HOLDOUTS = {3: 'stable_diffusion_v_1_4', 4: 'stable_diffusion_v_1_5',
            5: 'VQDM', 6: 'wukong'}


def main():
    destination = ROOT / 'v100_results_v2.tar.gz'
    with tarfile.open(destination, 'x:gz') as archive:
        archive.add(ROOT / 'INDEPENDENT_AUDIT.json', arcname='INDEPENDENT_AUDIT.json')
        for index, name in HOLDOUTS.items():
            folder = ROOT / f'task_14492182_{index}' / 'outputs' / name
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix in ('.csv', '.json'):
                    archive.add(path, arcname=(Path('outputs') / name / path.relative_to(folder)).as_posix())
    print(destination, destination.stat().st_size)


if __name__ == '__main__':
    main()
