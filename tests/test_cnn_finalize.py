"""CNN finalizer control-flow checks without model dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cnn_finalize", ROOT / "code/cnn/finalize_benchmark.py"
)
FINALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZE)


class CnnFinalizeTests(unittest.TestCase):
    def notebook(self, root: Path, complete: bool) -> Path:
        imports = '''from dataclasses import dataclass
class Device:
    type = "cpu"
DEVICE = Device()
if DEVICE.type != "cuda":
    raise RuntimeError("Select a GPU-backed Jupyter kernel before running the CNN benchmark.")
x = torch.ones((16, 16), device=DEVICE)
assert (x @ x).sum().item() == 4096
torch.cuda.synchronize()
print("GPU operation passed:", torch.cuda.get_device_name(0))
'''
        config = '''@dataclass
class Config:
    data_root: str = ""
    benchmark_root: str = ""
    experiment_root: str = ""
    output_root: str = ""
CFG = Config()
EXPERIMENT_PLAN_FILES = []
SELECTED_EXPERIMENTS = ["ALL"]
SESSION_TAG = "notebook"
'''
        training = "selected_runs = []\nraise AssertionError('finalizer entered training')\n"
        status = "complete" if complete else "incomplete"
        count = 192 if complete else 191
        audit = f'''def audit_completed_run(path):
    return None
audit_report = {{"status": {status!r}, "audited_runs": {count}}}
'''
        cells = [
            {"cell_type": "code", "source": source.splitlines(keepends=True)}
            for source in (imports, config, training, audit)
        ]
        path = root / "cnn.ipynb"
        path.write_text(json.dumps({"cells": cells}))
        return path

    def test_complete_audit_succeeds_and_training_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = FINALIZE.main([
                "--notebook", str(self.notebook(root, True)),
                "--shared-root", str(root),
            ])
            self.assertEqual(code, 0)

    def test_incomplete_audit_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = FINALIZE.main([
                "--notebook", str(self.notebook(root, False)),
                "--shared-root", str(root),
            ])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
