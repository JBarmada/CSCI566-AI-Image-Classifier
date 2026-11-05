#!/usr/bin/env python3
"""Execute the master notebook as a resumable, non-interactive batch job."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone


class StopForCheckpoint(BaseException):
    pass


def stop_requested(signum, frame):
    raise StopForCheckpoint(f'Scheduler signal {signum}; resume from the last saved epoch.')


def check_stale_lock(path):
    """Only archive a lock when the recorded process/job can be proved inactive."""
    record = json.loads(path.read_text())
    host, pid = record.get('host'), record.get('pid')
    if not host or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(f'Cannot verify malformed lock: {path}')
    if host.split('.')[0] == platform.node().split('.')[0]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        raise RuntimeError(f'Active or unverifiable process owns {path}: {record}')
    # Old notebook locks lack a Slurm ID. If *any* allocation remains on that
    # node, retain the lock. A scheduler/query failure also keeps it intact.
    query = ['squeue', '--noheader', '--nodes', host.split('.')[0], '--format', '%i']
    result = subprocess.run(query, capture_output=True, text=True, timeout=30, check=True)
    if result.stdout.strip():
        raise RuntimeError(f'Lock owner node still has active allocations; retaining {path}: {record}')


@contextlib.contextmanager
def study_lock(root):
    # flock is released by the OS after a hard kill; no manual lock removal needed.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'BATCH.lock').open('a+') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another batch writer is using {root}')
        yield


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from batch_session import apply_options, run_session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).resolve().parents[2] / 'ViT_Master_Experiments.ipynb')
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--study-name')
    parser.add_argument('--holdout', choices=['ADM', 'BigGAN', 'glide', 'Midjourney',
                        'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'])
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--recover-stale-locks', action='store_true')
    args = parser.parse_args(argv)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    nb = json.loads(args.notebook.read_text(encoding='utf-8'))
    code = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    if not code or not code[0][1].startswith('from pathlib import Path'):
        raise ValueError('Unexpected master notebook structure.')
    ns = {'__name__': '__main__', '__file__': str(args.notebook.resolve())}

    def execute(index, source):
        print(f'\n--- Notebook cell {index} ---', flush=True)
        exec(compile(source, f'<vit-master-cell-{index}>', 'exec'), ns)

    execute(*code[0])
    apply_options(ns, args)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    job_dir = ns['OUTPUT_ROOT'] / 'batch_jobs' / f'{os.environ.get("SLURM_JOB_ID", "manual")}_{stamp}'
    with study_lock(ns['OUTPUT_ROOT']):
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / 'input_notebook.ipynb').write_bytes(args.notebook.read_bytes())
        # Archive the source files behind the thin notebook, not just its wrapper cells.
        engine = args.notebook.parent / 'code/vit'
        for source in engine.rglob('*.py') if engine.is_dir() else []:
            destination = job_dir / 'source/code/vit' / source.relative_to(engine)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        return run_session(ns, code, execute, args, job_dir, stamp, check_stale_lock, StopForCheckpoint)


if __name__ == '__main__':
    sys.exit(main())
