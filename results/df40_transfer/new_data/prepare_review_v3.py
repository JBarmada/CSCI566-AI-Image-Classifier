"""Pre-score quarantine of two visually questionable candidate pairs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'fresh_candidate_review_v2'
DATA = ROOT / 'fresh_candidate_data_v1'
NEW = ROOT / 'fresh_candidate_review_v3'
EXCLUSIONS = {'113_983': 'Swap crops show a background portrait, not the selected real face.',
              '867_765': 'Wav2Lip crop appears unrelated to either selected real face; source role unverified.'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (OLD / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    removed = [r for r in rows if r['pair'] in EXCLUSIONS]
    retained = [r for r in rows if r['pair'] not in EXCLUSIONS]
    if len(removed) != 14 or len(retained) != 189 or len({r['sha256'] for r in retained}) != 189:
        raise ValueError('Unexpected quarantine counts or duplicate image hash')
    if any(digest(DATA / row['path']) != row['sha256'] for row in rows):
        raise ValueError('Candidate image hash mismatch')
    NEW.mkdir(exist_ok=False)
    with (NEW / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(retained)
    checksums = ''.join(r['sha256'] + '  ' + r['path'] + '\n' for r in retained)
    (NEW / 'checksums.sha256').write_bytes(checksums.encode())
    report = {'status': 'unscored exploratory candidate', 'prior_manifest_sha256': digest(OLD / 'manifest.csv'),
              'manifest_sha256': digest(NEW / 'manifest.csv'), 'excluded_pairs': EXCLUSIONS,
              'excluded_rows': len(removed), 'retained_pairs': 27, 'retained_rows': len(retained),
              'unique_hashes': len({r['sha256'] for r in retained}),
              'limits': 'Identity, source roles and mirror provenance remain unverified.'}
    (NEW / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
