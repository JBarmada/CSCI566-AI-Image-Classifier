"""Check candidate paths against local real and pinned remote ZIP indexes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from remote_zip import RemoteZipFile
from retrieve_adaptation import REAL, REV, SIZES as BASE_SIZES

SIZES = {**BASE_SIZES, 'inswap': 2502440755, 'pirender': 1763272422}


def check(method, rows):
    if method == 'real':
        source = ZipFile(REAL)
        names = [r['metadata_path'].removeprefix('deepfakes_detection_datasets/') for r in rows]
    else:
        url = f'https://huggingface.co/datasets/ManhQuangAI/DF40_train/resolve/{REV}/{method}.zip?download=true'
        source = ZipFile(RemoteZipFile(None, SIZES[method], url))
        marker = method + '/frames/'
        names = [marker + r['metadata_path'].split(marker, 1)[1] for r in rows]
    with source:
        listing = source.namelist()
        counts = {name: listing.count(name) for name in names}
    missing = [name for name, count in counts.items() if count == 0]
    duplicate = [name for name, count in counts.items() if count > 1]
    return {'requested': len(names), 'unique_paths': len(counts),
            'missing': missing, 'ambiguous': duplicate}


def main():
    plan = ROOT / 'fresh_candidate_plan_v1.json'
    rows = json.loads(plan.read_text())['samples']
    methods = sorted({row['method'] for row in rows})
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {m: pool.submit(check, m, [r for r in rows if r['method'] == m]) for m in methods}
        output = {m: job.result() for m, job in jobs.items()}
    report = {'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(), 'methods': output,
              'note': 'Index membership only; no image bytes, identity or license verified.'}
    (ROOT / 'fresh_candidate_member_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if any(v['missing'] or v['ambiguous'] or v['requested'] != v['unique_paths'] for v in output.values()):
        raise ValueError('Candidate membership check failed')


if __name__ == '__main__':
    main()
