"""Benchmark acceptance, isolated smoke manifests, and atomic progress reporting."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

_PROGRESS = {}


def validate_benchmark(data_root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
    from audit_benchmark import audit
    result = audit(data_root / 'benchmark')
    installed = data_root.parent.parent.parent / 'results/benchmark_acceptance.json'
    accepted = json.loads(installed.read_text())
    if (accepted.get('status') != 'passed' or not accepted.get('installed_image_bytes_verified')
            or accepted.get('fingerprint') != result['fingerprint']):
        raise ValueError('Installed benchmark has no matching successful image audit')
    return result['fingerprint']


def prepare_smoke(manifests, output, holdout):
    name = f'holdout_{holdout}_k1_rep0'
    destination = output / 'smoke_manifests' / name
    destination.mkdir(parents=True, exist_ok=True)
    for split, count in [('train', 32), ('val', 16), ('test', 16)]:
        with (manifests / name / f'{split}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        selected = [r for label in ['0', '1'] for r in
                    [row for row in rows if row['label'] == label][:count]]
        path = destination / f'{split}.csv'
        import io
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
        text = buffer.getvalue()
        if path.exists() and path.read_text() != text:
            raise ValueError('Existing smoke manifest differs; use a separate smoke study')
        if not path.exists():
            path.write_text(text)
    return destination.parent


def report_progress(root, state, force=False, **fields):
    key = str(root)
    if key not in _PROGRESS:
        initial = len(list(Path(root).glob('runs/*/evaluations/clean_*/COMPLETE.json')))
        _PROGRESS[key] = {'started_unix': time.time(), 'completed': initial,
                          'completed_at_start': initial}
    record = _PROGRESS[key]
    record.update(fields, state=state)
    if not force and time.time() - record.get('written_unix', 0) < 5:
        return
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record.update(updated_utc=datetime.now(timezone.utc).isoformat(), written_unix=time.time(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    temporary = root / f'progress.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, root / 'progress.json')
