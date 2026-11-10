#!/usr/bin/env python3
"""Run the independent CNN audit and fail unless all 192 runs pass."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sys


@contextlib.contextmanager
def audit_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "FINAL_AUDIT.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None) -> int:
    default_notebook = Path(__file__).resolve().parents[2] / "cnn.ipynb"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=default_notebook)
    parser.add_argument(
        "--shared-root", type=Path,
        default=Path("/project2/yzhao010_1531/csci566-genimage"),
    )
    args = parser.parse_args(argv)
    notebook = json.loads(args.notebook.resolve().read_text(encoding="utf-8"))
    code = [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]
    shared_root = args.shared_root.resolve()
    output_root = shared_root / "results/cnn_benchmark/notebook"
    ns = {
        "__name__": "__main__", "__file__": str(args.notebook.resolve()),
        "display": lambda value: print(value, flush=True),
    }
    os.environ["CNN_AUDIT_ONLY"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    configured = False
    audited = False
    with audit_lock(output_root):
        for index, source in code:
            stripped = source.lstrip()
            if stripped.startswith("# Run only if packages are missing"):
                continue
            if stripped.startswith("selected_runs ="):
                continue
            print(f"\n--- Notebook cell {index} ---", flush=True)
            exec(compile(source, f"<cnn-audit-cell-{index}>", "exec"), ns)
            if stripped.startswith("@dataclass"):
                data_root = shared_root / "data/extracted/GenImage_Compact"
                ns["CFG"].data_root = str(data_root)
                ns["CFG"].benchmark_root = str(data_root / "benchmark")
                ns["CFG"].experiment_root = str(data_root / "benchmark/manifests")
                ns["CFG"].output_root = str(output_root)
                ns["EXPERIMENT_PLAN_FILES"] = [
                    f"{ns['CFG'].benchmark_root}/experiments.json"
                ]
                ns["SELECTED_EXPERIMENTS"] = ["ALL"]
                ns["SESSION_TAG"] = "final_audit"
                configured = True
            if stripped.startswith("def audit_completed_run"):
                audited = True
                break
    if not configured or not audited:
        raise RuntimeError("Unexpected cnn.ipynb structure; audit did not execute")
    report = ns.get("audit_report", {})
    if report.get("status") != "complete" or report.get("audited_runs") != 192:
        print(f"CNN FINAL AUDIT FAILED: {report}", file=sys.stderr)
        return 1
    print("CNN FINAL AUDIT PASSED: 192/192 runs", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
