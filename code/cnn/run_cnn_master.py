#!/usr/bin/env python3
"""Execute cnn.ipynb as one resumable holdout batch job."""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback


HOLDOUTS = [
    "ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5", "VQDM", "wukong",
]
K_VALUES = (1, 2, 4, 7)
REPEATS = (0, 1, 2)
VARIANTS = ("baseline", "augmented")


class StopForCheckpoint(BaseException):
    pass


def stop_requested(signum, frame):
    raise StopForCheckpoint(
        f"Scheduler signal {signum}; resume from the last atomically saved epoch."
    )


def check_stale_lock(path: Path) -> None:
    """Return only when the recorded lock owner can be proved inactive."""
    record = json.loads(path.read_text())
    host, pid = record.get("host"), record.get("pid")
    if not host or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(f"Cannot verify malformed lock: {path}")
    if host.split(".")[0] == platform.node().split(".")[0]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        raise RuntimeError(f"Active or unverifiable process owns {path}: {record}")
    result = subprocess.run(
        ["squeue", "--noheader", "--nodes", host.split(".")[0], "--format", "%i"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            f"Lock owner node still has active allocations; retaining {path}: {record}"
        )


@contextlib.contextmanager
def holdout_lock(output_root: Path, holdout: str):
    """Use an OS-managed lock so a hard-killed job does not leave it held."""
    lock_dir = output_root / "batch_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{holdout}.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another CNN batch writer is running {holdout}") from exc
        yield


def notebook_code(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]


def configure_namespace(ns: dict, shared_root: Path, holdout: str) -> None:
    data_root = shared_root / "data/extracted/GenImage_Compact"
    ns["CFG"].data_root = str(data_root)
    ns["CFG"].benchmark_root = str(data_root / "benchmark")
    ns["CFG"].experiment_root = str(data_root / "benchmark/manifests")
    ns["CFG"].output_root = str(shared_root / "results/cnn_benchmark/notebook")
    ns["CFG"].skip_completed = True
    ns["CFG"].resume = True
    ns["CFG"].continue_on_error = False
    ns["CFG"].max_runs_this_session = None
    ns["CFG"].debug_max_rows_per_split = None
    ns["EXPERIMENT_PLAN_FILES"] = [f"{ns['CFG'].benchmark_root}/experiments.json"]
    ns["SELECTED_EXPERIMENTS"] = [
        f"holdout_{holdout}_k{k}_rep{repeat}"
        for k in K_VALUES for repeat in REPEATS
    ]
    ns["TRAINING_VARIANTS"] = list(VARIANTS)
    ns["SESSION_TAG"] = f"holdout_{holdout}"


def selected_run_directories(output_root: Path, holdout: str):
    for k in K_VALUES:
        for repeat in REPEATS:
            experiment = f"holdout_{holdout}_k{k}_rep{repeat}"
            for variant in VARIANTS:
                yield output_root / "runs" / f"{experiment}__{variant}__seed17"


def status_writer(job_dir: Path, args, output_root: Path, stamp: str):
    revision = subprocess.run(
        ["git", "-C", str(args.notebook.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    status = {
        "state": "running", "started_utc": stamp, "holdout": args.holdout,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
        "python": sys.executable, "argv": sys.argv,
        "notebook": str(args.notebook.resolve()), "output_root": str(output_root),
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
    }

    def save(state: str, **extra) -> None:
        status.update(state=state, **extra)
        temporary = job_dir / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(temporary, job_dir / "status.json")

    return save


def execute_notebook(ns: dict, code: list[tuple[int, str]], args, shared_root: Path,
                     recover_stamp: str) -> None:
    configured = False
    for index, source in code:
        stripped = source.lstrip()
        if stripped.startswith("# Run only if packages are missing"):
            continue
        if stripped.startswith("def audit_completed_run"):
            break
        if stripped.startswith("selected_runs ="):
            if args.preflight_only:
                print("Preflight passed; no training requested.", flush=True)
                return
            if args.recover_stale_locks:
                for run_dir in selected_run_directories(
                    Path(ns["CFG"].output_root), args.holdout
                ):
                    lock = run_dir / "RUNNING.lock"
                    if lock.exists():
                        check_stale_lock(lock)
                        lock.rename(lock.with_name(f"RUNNING.lock.stale.{recover_stamp}"))
        print(f"\n--- Notebook cell {index} ---", flush=True)
        exec(compile(source, f"<cnn-master-cell-{index}>", "exec"), ns)
        if stripped.startswith("@dataclass"):
            configure_namespace(ns, shared_root, args.holdout)
            configured = True
    if not configured:
        raise ValueError("Could not locate the CNN notebook configuration cell.")
    if args.preflight_only:
        return
    failed = [row for row in ns.get("session_rows", []) if row["status"].startswith("failed:")]
    unfinished = [
        str(path) for path in selected_run_directories(Path(ns["CFG"].output_root), args.holdout)
        if not (path / "TRAINING_COMPLETE.json").is_file()
    ]
    if failed or unfinished:
        raise RuntimeError(
            f"Incomplete holdout: {len(failed)} failures and {len(unfinished)} unfinished runs"
        )


def main(argv=None) -> int:
    default_notebook = Path(__file__).resolve().parents[2] / "cnn.ipynb"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=default_notebook)
    parser.add_argument(
        "--shared-root", type=Path,
        default=Path("/project2/yzhao010_1531/csci566-genimage"),
    )
    parser.add_argument("--holdout", required=True, choices=HOLDOUTS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recover-stale-locks", action="store_true")
    args = parser.parse_args(argv)
    args.notebook = args.notebook.resolve()
    shared_root = args.shared_root.resolve()
    output_root = shared_root / "results/cnn_benchmark/notebook"
    code = notebook_code(args.notebook)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    job_dir = output_root / "batch_jobs" / args.holdout / f"{job_id}_{stamp}"
    ns = {
        "__name__": "__main__", "__file__": str(args.notebook),
        "display": lambda value: print(value, flush=True),
    }
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        with holdout_lock(output_root, args.holdout):
            job_dir.mkdir(parents=True, exist_ok=False)
            (job_dir / "input_notebook.ipynb").write_bytes(args.notebook.read_bytes())
            source_root = Path(__file__).resolve().parent
            for source in source_root.glob("*"):
                if source.suffix in {".py", ".slurm"}:
                    destination = job_dir / "source/code/cnn" / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())
            save = status_writer(job_dir, args, output_root, stamp)
            save("running")
            if hasattr(signal, "SIGUSR1"):
                signal.signal(signal.SIGUSR1, stop_requested)
            signal.signal(signal.SIGTERM, stop_requested)
            try:
                execute_notebook(ns, code, args, shared_root, stamp)
                state = "preflight_passed" if args.preflight_only else "complete"
                save(state, selected_runs=0 if args.preflight_only else 24,
                     finished_utc=datetime.now(timezone.utc).isoformat())
                return 0
            except (StopForCheckpoint, KeyboardInterrupt) as error:
                save("interrupted_resumable", error=str(error))
                print(f"{error}\nSubmit the same array index again to resume.", flush=True)
                return 75
            except Exception as error:
                save("failed", error=repr(error), traceback=traceback.format_exc())
                traceback.print_exc()
                return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
