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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).with_name('ViT_Master_Experiments.ipynb'))
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--study-name')
    parser.add_argument('--preflight-only', action='store_true', help='Validate all manifests/protocols without training.')
    parser.add_argument('--recover-stale-locks', action='store_true', help='Archive only locks proved inactive by process/scheduler checks.')
    args = parser.parse_args(argv)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    nb = json.loads(args.notebook.read_text(encoding='utf-8'))
    code = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    if not code or not code[0][1].startswith('from pathlib import Path'):
        raise ValueError('Unexpected master notebook structure.')
    namespace = {'__name__': '__main__', '__file__': str(args.notebook.resolve())}

    def execute(index, source):
        print(f'\n--- Notebook cell {index} ---', flush=True)
        # Compile each cell separately, just as Jupyter does. Do not regenerate
        # training functions in a separate imported module (different fingerprints).
        exec(compile(source, f'<vit-master-cell-{index}>', 'exec'), namespace)

    execute(*code[0])
    if args.shared_root:
        namespace['SHARED_ROOT'] = args.shared_root
        namespace['DATA_ROOT'] = args.shared_root / 'data/extracted/GenImage_Compact'
        namespace['MANIFEST_ROOT'] = namespace['DATA_ROOT'] / '_experiment_manifests'
    if args.study_name:
        namespace['STUDY_NAME'] = args.study_name
    namespace['OUTPUT_ROOT'] = namespace['SHARED_ROOT'] / 'results' / namespace['STUDY_NAME']
    namespace['FEATURE_CACHE_ROOT'] = namespace['SHARED_ROOT'] / 'features' / namespace['STUDY_NAME']
    # Finish all selected runs, ignoring an interactive session's optional work cap.
    namespace['MAX_RUNS_THIS_SESSION'] = None
    namespace['CONTINUE_ON_ERROR'] = False
    root = namespace['OUTPUT_ROOT']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    job_dir = root / 'batch_jobs' / f'{os.environ.get("SLURM_JOB_ID", "manual")}_{stamp}'

    with study_lock(root):
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / 'input_notebook.ipynb').write_bytes(args.notebook.read_bytes())
        status = {'state': 'running', 'started_utc': stamp,
                  'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': platform.node(),
                  'python': sys.executable, 'argv': sys.argv,
                  'notebook': str(args.notebook.resolve()), 'output_root': str(root)}

        def save_status(state, **extra):
            status.update(state=state, **extra)
            tmp = job_dir / 'status.tmp'
            tmp.write_text(json.dumps(status, indent=2), encoding='utf-8')
            os.replace(tmp, job_dir / 'status.json')

        save_status('running')
        print('Batch status folder:', job_dir, flush=True)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        summary = None
        try:
            for index, source in code[1:]:
                if source.startswith('session_rows ='):
                    if args.preflight_only:
                        print('Preflight passed; no training/evaluation requested.', flush=True)
                        save_status('preflight_passed', selected_runs=len(namespace['PLAN']))
                        return 0
                    # Preflight reconciles *all* existing configs before any training.
                    if args.recover_stale_locks:
                        for entry in namespace['PLAN']:
                            lock = entry['directory'] / 'RUNNING.lock'
                            if lock.exists():
                                check_stale_lock(lock)
                                destination = lock.with_name(f'RUNNING.lock.stale.{stamp}')
                                lock.rename(destination)
                                print('Archived verified stale lock:', destination, flush=True)
                if source.startswith('rows, missing ='):
                    summary = (index, source)
                execute(index, source)
            failed = [row for row in namespace.get('session_rows', []) if row['status'].startswith('failed:')]
            unfinished_training = [str(e['directory']) for e in namespace['PLAN']
                                   if not (e['directory'] / 'TRAINING_COMPLETE.json').exists()]
            missing_evaluations = namespace.get('missing', []) if namespace['RUN_FINAL_TEST'] else []
            if failed or unfinished_training or missing_evaluations:
                raise RuntimeError(f'Incomplete study: {len(failed)} failures, '
                                   f'{len(unfinished_training)} untrained, '
                                   f'{len(missing_evaluations)} missing evaluations.')
            save_status('complete', selected_runs=len(namespace['PLAN']),
                        summary_dir=str(namespace.get('SUMMARY_DIR', '')),
                        finished_utc=datetime.now(timezone.utc).isoformat())
            print('\nALL SELECTED RUNS COMPLETE. Summaries:', namespace.get('SUMMARY_DIR'), flush=True)
            return 0
        except (StopForCheckpoint, KeyboardInterrupt) as error:
            save_status('interrupted_resumable', error=str(error))
            print(str(error), flush=True)
            print('Submit the same Slurm job again to resume.', flush=True)
            return 75
        except Exception as error:
            save_status('failed', error=repr(error), traceback=traceback.format_exc())
            traceback.print_exc()
            # Keep already-complete results available even if a later run fails.
            if summary is None:
                summary = next(((i, s) for i, s in code if s.startswith('rows, missing =')), None)
            if summary and 'PLAN' in namespace:
                try:
                    execute(*summary)
                except Exception:
                    traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())
