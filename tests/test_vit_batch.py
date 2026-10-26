"""Batch control-flow checks; no Slurm allocation, CUDA or real images required."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vit_batch', ROOT / 'code/vit/run_vit_master.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@contextlib.contextmanager
def no_flock(root):
    yield


class BatchTests(unittest.TestCase):
    def notebook(self, root, outcome='complete'):
        settings = f'''from pathlib import Path
SHARED_ROOT = Path({str(root)!r})
STUDY_NAME = 'study'
MAX_RUNS_THIS_SESSION = 1
CONTINUE_ON_ERROR = True
RUN_FINAL_TEST = True
'''
        plan = "PLAN = [{'directory': OUTPUT_ROOT / 'runs' / 'one'}]\n"
        runner = '''session_rows = []
assert MAX_RUNS_THIS_SESSION is None
assert CONTINUE_ON_ERROR is False
directory = PLAN[0]['directory']
directory.mkdir(parents=True)
'''
        if outcome == 'complete':
            runner += "(directory / 'TRAINING_COMPLETE.json').write_text('{}')\nsession_rows.append({'status': 'complete'})\n"
        elif outcome == 'interrupted':
            runner += "raise KeyboardInterrupt('synthetic scheduler interruption')\n"
        elif outcome == 'failed':
            runner += "raise ValueError('synthetic training failure')\n"
        summary = "rows, missing = [], []\nSUMMARY_DIR = OUTPUT_ROOT / 'summaries'\nSUMMARY_DIR.mkdir(exist_ok=True)\n"
        cells = [{'cell_type': 'code', 'source': s.splitlines(keepends=True)}
                 for s in [settings, plan, runner, summary]]
        path = root / 'notebook.ipynb'
        path.write_text(json.dumps({'cells': cells}))
        return path

    def run_case(self, outcome, preflight=False):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notebook = self.notebook(root, outcome)
            arguments = ['--notebook', str(notebook)]
            if preflight:
                arguments.append('--preflight-only')
            with patch.object(batch, 'study_lock', no_flock), patch.object(batch.signal, 'signal'):
                exit_code = batch.main(arguments)
            statuses = list((root / 'results/study/batch_jobs').glob('*/status.json'))
            self.assertEqual(len(statuses), 1)
            return exit_code, json.loads(statuses[0].read_text())

    def test_complete_status(self):
        code, status = self.run_case('complete')
        self.assertEqual(code, 0)
        self.assertEqual(status['state'], 'complete')

    def test_preflight_never_trains(self):
        code, status = self.run_case('failed', preflight=True)
        self.assertEqual(code, 0)
        self.assertEqual(status['state'], 'preflight_passed')

    def test_default_notebook_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notebook = self.notebook(root, 'failed')
            notebook = notebook.rename(root / 'ViT_Master_Experiments.ipynb')
            script = root / 'code/vit/run_vit_master.py'
            script.parent.mkdir(parents=True)
            script.touch()
            elsewhere = root / 'elsewhere'
            elsewhere.mkdir()
            original_cwd = Path.cwd()
            try:
                os.chdir(elsewhere)
                with patch.object(batch, '__file__', str(script)), \
                     patch.object(batch, 'study_lock', no_flock), \
                     patch.object(batch.signal, 'signal'):
                    code = batch.main(['--preflight-only'])
            finally:
                os.chdir(original_cwd)
            statuses = list((root / 'results/study/batch_jobs').glob('*/status.json'))
            self.assertEqual(code, 0)
            self.assertEqual(len(statuses), 1)
            status = json.loads(statuses[0].read_text())
            self.assertEqual(status['state'], 'preflight_passed')
            self.assertEqual(status['notebook'], str(notebook.resolve()))

    def test_interruption_is_resumable_not_success(self):
        code, status = self.run_case('interrupted')
        self.assertEqual(code, 75)
        self.assertEqual(status['state'], 'interrupted_resumable')

    def test_failure_is_not_success(self):
        code, status = self.run_case('failed')
        self.assertEqual(code, 1)
        self.assertEqual(status['state'], 'failed')

    def test_missing_training_is_not_success(self):
        code, status = self.run_case('missing')
        self.assertEqual(code, 1)
        self.assertEqual(status['state'], 'failed')

    def test_remote_active_lock_is_retained(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'RUNNING.lock'
            path.write_text(json.dumps({'host': 'old-node', 'pid': 123}))
            with patch.object(batch.platform, 'node', return_value='new-node'), \
                 patch.object(batch.subprocess, 'run') as run:
                run.return_value.stdout = '123456\n'
                with self.assertRaisesRegex(RuntimeError, 'still has active allocations'):
                    batch.check_stale_lock(path)
            self.assertTrue(path.exists())

    def test_remote_inactive_lock_is_not_deleted_by_check(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'RUNNING.lock'
            path.write_text(json.dumps({'host': 'old-node', 'pid': 123}))
            with patch.object(batch.platform, 'node', return_value='new-node'), \
                 patch.object(batch.subprocess, 'run') as run:
                run.return_value.stdout = ''
                batch.check_stale_lock(path)
            self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
