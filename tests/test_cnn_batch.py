"""CNN batch control-flow checks without CUDA or real images."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cnn_batch", ROOT / "code/cnn/run_cnn_master.py"
)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


@contextlib.contextmanager
def no_lock(output_root, holdout):
    yield


class CnnBatchTests(unittest.TestCase):
    def notebook(self, root: Path, outcome: str) -> Path:
        imports = """from dataclasses import dataclass
from pathlib import Path
from typing import Optional
"""
        config = f'''@dataclass
class Config:
    data_root: str = ""
    benchmark_root: str = ""
    experiment_root: str = ""
    output_root: str = ""
    skip_completed: bool = False
    resume: bool = False
    continue_on_error: bool = True
    max_runs_this_session: Optional[int] = 1
    debug_max_rows_per_split: Optional[int] = 1
CFG = Config()
EXPERIMENT_PLAN_FILES = []
SELECTED_EXPERIMENTS = ["ALL"]
TRAINING_VARIANTS = ["baseline"]
SESSION_TAG = "notebook"
'''
        plan = "selected_experiments = [{'experiment': name} for name in SELECTED_EXPERIMENTS]\n"
        training = """selected_runs = [(entry, variant) for entry in selected_experiments for variant in TRAINING_VARIANTS]
session_rows = []
assert CFG.resume and CFG.skip_completed
assert CFG.max_runs_this_session is None and not CFG.continue_on_error
assert SESSION_TAG.startswith('holdout_')
for entry, variant in selected_runs:
    directory = Path(CFG.output_root) / 'runs' / f"{entry['experiment']}__{variant}__seed17"
    directory.mkdir(parents=True, exist_ok=True)
"""
        if outcome == "complete":
            training += "    (directory / 'TRAINING_COMPLETE.json').write_text('{}')\n    session_rows.append({'status': 'complete'})\n"
        elif outcome == "interrupted":
            training += "    raise KeyboardInterrupt('synthetic interruption')\n"
        elif outcome == "failed":
            training += "    raise ValueError('synthetic failure')\n"
        elif outcome == "missing":
            training += "    session_rows.append({'status': 'complete'})\n"
        audit = "def audit_completed_run(path):\n    raise AssertionError('worker executed audit')\n"
        cells = [
            {"cell_type": "code", "source": source.splitlines(keepends=True)}
            for source in (imports, config, plan, training, audit)
        ]
        path = root / "cnn.ipynb"
        path.write_text(json.dumps({"cells": cells}))
        return path

    def run_case(self, outcome: str, preflight: bool = False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = self.notebook(root, outcome)
            arguments = [
                "--notebook", str(notebook), "--shared-root", str(root),
                "--holdout", "ADM",
            ]
            if preflight:
                arguments.append("--preflight-only")
            with patch.object(BATCH, "holdout_lock", no_lock), \
                 patch.object(BATCH.signal, "signal"):
                code = BATCH.main(arguments)
            statuses = list((root / "results/cnn_benchmark/notebook/batch_jobs/ADM").glob(
                "*/status.json"
            ))
            self.assertEqual(len(statuses), 1)
            return code, json.loads(statuses[0].read_text())

    def test_complete_holdout(self):
        code, status = self.run_case("complete")
        self.assertEqual(code, 0)
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["selected_runs"], 24)

    def test_preflight_does_not_train(self):
        code, status = self.run_case("failed", preflight=True)
        self.assertEqual(code, 0)
        self.assertEqual(status["state"], "preflight_passed")

    def test_interruption_is_resumable(self):
        code, status = self.run_case("interrupted")
        self.assertEqual(code, 75)
        self.assertEqual(status["state"], "interrupted_resumable")

    def test_failure_is_not_success(self):
        code, status = self.run_case("failed")
        self.assertEqual(code, 1)
        self.assertEqual(status["state"], "failed")

    def test_missing_outputs_are_not_success(self):
        code, status = self.run_case("missing")
        self.assertEqual(code, 1)
        self.assertEqual(status["state"], "failed")

    def test_remote_active_lock_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "RUNNING.lock"
            lock.write_text(json.dumps({"host": "old-node", "pid": 123}))
            with patch.object(BATCH.platform, "node", return_value="new-node"), \
                 patch.object(BATCH.subprocess, "run") as run:
                run.return_value.stdout = "123456\n"
                with self.assertRaisesRegex(RuntimeError, "still has active allocations"):
                    BATCH.check_stale_lock(lock)
            self.assertTrue(lock.exists())


if __name__ == "__main__":
    unittest.main()
