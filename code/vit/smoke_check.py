"""GPU smoke test: interrupt/resume both modes, then verify idempotent completion."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SHARED = Path('/project2/yzhao010_1531/csci566-genimage')
ROOT = Path(__file__).resolve().parents[2]


def run(extra, expected, resume=False):
    command = [sys.executable, '-u', str(ROOT / 'code/vit/run_vit_master.py'),
               '--holdout', 'ADM', '--smoke-test', *extra]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout, flush=True)
    print(result.stderr, file=sys.stderr, flush=True)
    if result.returncode != expected:
        raise RuntimeError(f'Smoke step returned {result.returncode}, expected {expected}')
    if resume and 'Resuming after epoch 1' not in result.stdout:
        raise RuntimeError('Expected checkpoint resume did not occur')


def validate():
    runs = SHARED / 'results/vit_benchmark_smoke/ADM/runs'
    hashes = {}
    for mode in ['baseline', 'augmented']:
        folder = runs / f'holdout_ADM_k1_rep0__{mode}__seed17'
        config = json.loads((folder / 'config.json').read_text())
        if not config['smoke_test'] or config['max_epochs'] != 2:
            raise ValueError('Smoke outputs mixed with full study')
        with (folder / 'training_history.csv').open() as stream:
            if [int(r['epoch']) for r in csv.DictReader(stream)] != [1, 2]:
                raise ValueError('Resume duplicated or omitted epochs')
        marker = json.loads((folder / 'TRAINING_COMPLETE.json').read_text())
        digest = hashlib.sha256((folder / 'best_head.pt').read_bytes()).hexdigest()
        if marker['checkpoint_sha256'] != digest:
            raise ValueError('Checkpoint differs from completion marker')
        predictions = list(folder.glob('evaluations/clean_*/test_predictions.csv'))
        if len(predictions) != 1:
            raise ValueError('Expected one clean evaluation')
        with predictions[0].open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 32 or not all(0 <= float(r['prob_fake']) <= 1 for r in rows):
            raise ValueError('Invalid test predictions')
        hashes[mode] = digest
    return hashes


def main():
    # A fresh smoke study is intentional: do not silently skip the interruption tests.
    target = SHARED / 'results/vit_benchmark_smoke/ADM/runs'
    if target.exists() and list(target.iterdir()):
        raise FileExistsError('Smoke study already has runs; inspect them before repeating this test')
    run(['--stop-after-epoch', '1'], 75)
    run(['--stop-after-epoch', '1'], 75, resume=True)
    run([], 0, resume=True)
    before = validate()
    run([], 0)
    if before != validate():
        raise ValueError('Completed-run rerun changed checkpoints')
    report = SHARED / 'results/vit_benchmark_smoke/acceptance.json'
    report.write_text(json.dumps({'status': 'passed', 'both_modes_resumed': True,
                                 'completed_rerun_unchanged': True, 'checkpoints': before}, indent=2))
    print('GPU SMOKE TEST PASSED:', report, flush=True)


if __name__ == '__main__':
    main()
