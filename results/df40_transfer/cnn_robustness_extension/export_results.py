"""Archive CNN robustness result files and diagnostics without source images."""
import tarfile
from pathlib import Path

ROOT = Path('/project2/yzhao010_1531/csci566-genimage/results')
EXT = ROOT / 'cnn_robustness_extension_v1'
ADM = ROOT / 'cnn_robustness_k7_v1'
OUT = EXT / 'robustness_results_v1.tar.gz'
SUFFIXES = {'.json', '.csv'}


def add_tree(archive, directory, prefix):
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix in SUFFIXES and '.ipynb_checkpoints' not in path.parts:
            archive.add(path, arcname=(Path(prefix) / path.relative_to(directory)).as_posix())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    with tarfile.open(OUT, 'x:gz') as archive:
        add_tree(archive, EXT / 'outputs', 'extension')
        add_tree(archive, ADM, 'adm')
        for name in ('AUDIT_INVENTORY.json', 'run_notebook.py', 'audit.py', 'audit_inventory.py',
                     'CNN_K7_Robustness_Evaluation.ipynb'):
            archive.add(EXT / name, arcname=name)
    print('Saved', OUT, OUT.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
