"""Read original job logs and saved scores without changing benchmark data."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/project2/yzhao010_1531/csci566-genimage')


def main():
    logs = []
    for path in sorted(ROOT.glob('cnn-benchmark-13960524_*.log')):
        raw = path.read_bytes()
        text = raw.decode('utf-8', errors='replace')
        gpu = re.search(r'GPU operation passed: ([^\r\n]+)', text)
        torch = re.search(r'PyTorch: ([^\r\n]+)', text)
        holdouts = sorted(set(re.findall(r'^\s*-\s+holdout_(.+?)_k[1247]_rep[012]\s*$', text, re.MULTILINE)))
        logs.append({'name': path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'gpu': gpu.group(1) if gpu else None,
                     'pytorch': torch.group(1) if torch else None, 'holdouts_seen': holdouts})
    run = ROOT / 'results/cnn_benchmark/notebook/runs/holdout_stable_diffusion_v_1_4_k7_rep0__baseline__seed17'
    paths = ('config.json', 'test_predictions.csv', 'TRAINING_COMPLETE.json')
    times = {name: datetime.fromtimestamp((run / name).stat().st_mtime, timezone.utc).isoformat()
             for name in paths}
    result = {'logs': logs, 'example_file_mtime_utc': times,
              'example_marker': json.loads((run / 'TRAINING_COMPLETE.json').read_text())}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
