"""Exclude the whole affected pair after a score-blind duplicate-byte check."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'fresh_candidate_plan_v1.json'
DATA = ROOT / 'fresh_candidate_data_v1'
PAIR = '628_568'


def main():
    planned = json.loads(PLAN.read_text())['samples']
    by_key = {(row['method'], row['group']): row for row in planned}
    with (DATA / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    matched = [row for row in rows if by_key[(row['method'], row['video_group'])]['pair'] == PAIR]
    duplicate = [row for row in matched if row['method'] in ('simswap', 'inswap')]
    if len(matched) != 7 or len(duplicate) != 2 or duplicate[0]['sha256'] != duplicate[1]['sha256']:
        raise ValueError('Expected duplicate pair evidence changed')
    kept = [row for row in rows if row not in matched]
    if len(kept) != 203 or len({row['sha256'] for row in kept}) != 203:
        raise ValueError('Unexpected retained size or duplicate bytes')
    out = ROOT / 'fresh_candidate_review_v2'
    out.mkdir(exist_ok=False)
    with (out / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pair', *rows[0].keys()])
        writer.writeheader()
        for row in kept:
            writer.writerow({'pair': by_key[(row['method'], row['video_group'])]['pair'], **row})
    report = {'status': 'candidate pending provenance and protocol review; no scores computed',
              'original_plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
              'excluded_pair': PAIR, 'excluded_rows': len(matched),
              'duplicate_sha256': duplicate[0]['sha256'], 'retained_rows': len(kept),
              'retained_pairs': 29, 'unique_hashes': 203,
              'images_relative_to': '../fresh_candidate_data_v1/',
              'limits': 'Identities, source roles, license and mirror bytes remain unverified.'}
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
