"""Batch execution helpers, kept separate from CLI and locking."""
from datetime import datetime, timezone
import json
import os
import platform
import sys
import subprocess
import traceback


def apply_options(ns, args):
    if args.shared_root:
        ns['SHARED_ROOT'] = args.shared_root
        ns['DATA_ROOT'] = args.shared_root / 'data/extracted/GenImage_Compact'
        ns['MANIFEST_ROOT'] = ns['DATA_ROOT'] / 'benchmark/manifests'
    if args.holdout:
        ns['HOLDOUTS'] = [args.holdout]
        ns['STUDY_NAME'] = 'vit_benchmark/' + args.holdout
    if args.smoke_test:
        if not args.holdout:
            raise ValueError('--smoke-test requires --holdout')
        ns['SMOKE_TEST'] = True
        ns['STUDY_NAME'] = 'vit_benchmark_smoke/' + args.holdout
    if args.stop_after_epoch and not args.smoke_test:
        raise ValueError('--stop-after-epoch is only allowed in isolated smoke tests')
    ns['STOP_AFTER_EPOCH'] = args.stop_after_epoch
    if args.study_name:
        ns['STUDY_NAME'] = args.study_name
    ns['OUTPUT_ROOT'] = ns['SHARED_ROOT'] / 'results' / ns['STUDY_NAME']
    ns['FEATURE_CACHE_ROOT'] = ns['SHARED_ROOT'] / 'features' / ns['STUDY_NAME']
    ns['MAX_RUNS_THIS_SESSION'], ns['CONTINUE_ON_ERROR'] = None, False


def status_writer(ns, args, job_dir, stamp):
    revision = subprocess.run(['git', '-C', str(args.notebook.parent), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False)
    status = {'state': 'running', 'started_utc': stamp,
              'code_revision': revision.stdout.strip() if revision.returncode == 0 else None,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': platform.node(),
              'python': sys.executable, 'argv': sys.argv,
              'notebook': str(args.notebook.resolve()), 'output_root': str(ns['OUTPUT_ROOT'])}

    def save(state, **extra):
        status.update(state=state, **extra)
        temporary = job_dir / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        os.replace(temporary, job_dir / 'status.json')
        if 'report_progress' in ns:
            ns['report_progress'](ns['OUTPUT_ROOT'], state, force=True, **extra)

    return save


def execute_cells(ns, code, execute, args, stamp, check_lock, save):
    for index, source in code[1:]:
        if source.startswith('session_rows ='):
            if args.preflight_only:
                print('Preflight passed; no training/evaluation requested.', flush=True)
                save('preflight_passed', selected_runs=len(ns['PLAN']))
                return True
            if args.recover_stale_locks:
                for entry in ns['PLAN']:
                    lock = entry['directory'] / 'RUNNING.lock'
                    if lock.exists():
                        check_lock(lock)
                        lock.rename(lock.with_name(f'RUNNING.lock.stale.{stamp}'))
        execute(index, source)
    return False


def verify_completion(ns):
    failed = [r for r in ns.get('session_rows', []) if r['status'].startswith('failed:')]
    unfinished = [e for e in ns['PLAN'] if not (e['directory'] / 'TRAINING_COMPLETE.json').exists()]
    missing = ns.get('missing', []) if ns['RUN_FINAL_TEST'] else []
    if failed or unfinished or missing:
        raise RuntimeError(f'Incomplete study: {len(failed)} failures, {len(unfinished)} untrained, '
                           f'{len(missing)} missing evaluations')


def run_session(ns, code, execute, args, job_dir, stamp, check_lock, stop_type):
    save = status_writer(ns, args, job_dir, stamp)
    save('running')
    print('Batch status folder:', job_dir, flush=True)
    try:
        if execute_cells(ns, code, execute, args, stamp, check_lock, save):
            return 0
        verify_completion(ns)
        save('complete', selected_runs=len(ns['PLAN']), completed=len(ns['PLAN']),
             summary_dir=str(ns.get('SUMMARY_DIR', '')),
             finished_utc=datetime.now(timezone.utc).isoformat())
        print('ALL SELECTED RUNS COMPLETE. Summaries:', ns.get('SUMMARY_DIR'), flush=True)
        return 0
    except (stop_type, KeyboardInterrupt) as error:
        save('interrupted_resumable', error=str(error))
        print(f'{error}\nSubmit the same job again to resume.', flush=True)
        return 75
    except Exception as error:
        save('failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
