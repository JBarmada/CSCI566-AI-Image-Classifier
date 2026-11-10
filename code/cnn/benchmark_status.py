#!/usr/bin/env python3
"""Report CNN array progress without taking writer locks."""
import argparse
import json
from pathlib import Path
import time


HOLDOUTS = [
    "ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5", "VQDM", "wukong",
]


def report(root: Path) -> int:
    total = 0
    for holdout in HOLDOUTS:
        completed = len(list((root / "runs").glob(
            f"holdout_{holdout}_k*_rep*__*__seed17/TRAINING_COMPLETE.json"
        )))
        total += completed
        statuses = sorted((root / "batch_jobs" / holdout).glob("*/status.json"))
        state = "not started"
        if statuses:
            record = json.loads(statuses[-1].read_text())
            state = record.get("state", "unknown")
        progress_path = root / "sessions" / f"holdout_{holdout}" / "progress.json"
        detail = ""
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            age = time.time() - progress_path.stat().st_mtime
            detail = f" | {progress.get('run', progress.get('status', ''))} | update {age:.0f}s ago"
        print(f"{holdout}: {completed}/24 | {state}{detail}")
    print(f"Completed CNN runs: {total}/192")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/notebook"),
    )
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        report(args.root)
        if not args.watch:
            return
        time.sleep(30)
        print()


if __name__ == "__main__":
    main()
