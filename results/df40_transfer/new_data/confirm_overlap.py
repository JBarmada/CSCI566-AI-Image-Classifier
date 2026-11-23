"""Audit exact and inferred relationships to the earlier evaluation."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def tokens(value):
    return set(re.findall(r'(?<![A-Za-z0-9])id\d+(?!\d)', value))


def main():
    old = read(ROOT.parent / 'shared_dataset_v1/manifest.csv')
    new = read(ROOT / 'fake_candidates_v1/manifest.csv')
    old_hashes = {r['sha256'] for r in old}
    old_groups = {r['video_group'] for r in old}
    old_ids = set().union(*(tokens(g) for g in old_groups))
    counts = {}
    rows = []
    for row in new:
        shared = tokens(row['video_group']) & old_ids
        rows.append({'method': row['method'], 'video_group': row['video_group'],
                     'same_image_hash': row['sha256'] in old_hashes,
                     'same_fake_group_name': row['video_group'] in old_groups,
                     'shared_id_tokens': '|'.join(sorted(shared)),
                     'possible_prior_clip_overlap': row['possible_prior_clip_overlap']})
    for method in ('simswap', 'fomm', 'wav2lip'):
        group = [r for r in rows if r['method'] == method]
        counts[method] = {'images': len(group),
                          'exact_image_hash_matches': sum(r['same_image_hash'] for r in group),
                          'exact_group_name_matches': sum(r['same_fake_group_name'] for r in group),
                          'shared_identity_tokens': sum(bool(r['shared_id_tokens']) for r in group),
                          'possible_clip_matches': sum(bool(r['possible_prior_clip_overlap']) for r in group)}
    out = ROOT / 'fake_candidates_v1'
    with (out / 'overlap_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'overlap_review.json').write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
