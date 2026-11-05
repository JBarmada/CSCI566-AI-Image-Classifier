"""Show aggregate progress without starting training or taking writer locks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def snapshot(root):
    complete, failed = 0, 0
    lines, estimates = [], []
    for holdout in ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
                    'stable_diffusion_v_1_5', 'VQDM', 'wukong']:
        folder = root / holdout
        done = len(list(folder.glob('runs/*/evaluations/clean_*/COMPLETE.json')))
        complete += done
        progress = folder / 'progress.json'
        if not progress.exists():
            lines.append(f'{holdout}: {done}/24 | not started')
            continue
        record = json.loads(progress.read_text())
        state = record['state']
        failed += int(state == 'failed')
        age = time.time() - record.get('written_unix', time.time())
        detail = f"{record.get('experiment', '')} {record.get('mode', '')} epoch {record.get('epoch', '-')}"
        if record.get('batches'):
            detail += f" | batches {record.get('batch')}/{record['batches']}"
        lines.append(f'{holdout}: {done}/24 | {state} | {detail} | update {age:.0f}s ago')
        newly_done = done - record.get('completed_at_start', 0)
        if newly_done > 0 and state == 'running':
            elapsed = time.time() - record['started_unix']
            estimates.append((24 - done) * elapsed / newly_done)
    print(datetime.now(timezone.utc).isoformat())
    print(f'Completed evaluations: {complete}/192 | failed holdout jobs: {failed}')
    for line in lines:
        print(line)
    if estimates:
        print(f'Active-job remaining estimates: {", ".join(f"{s/3600:.1f}h" for s in estimates)}; '
              'rough estimates, excluding queued jobs.')
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(
        '/project2/yzhao010_1531/csci566-genimage/results/vit_benchmark'))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        done = snapshot(args.root)
        if not args.watch or done == 192:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
