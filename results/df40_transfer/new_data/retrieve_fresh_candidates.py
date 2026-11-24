"""Retrieve the preselected candidate crops without evaluating any models."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import retrieve_adaptation as original
from audit_candidate_members import SIZES
from verify_real_archive import crop

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'fresh_candidate_data_v1'


def existing_hashes():
    hashes = set()
    for path in ROOT.parent.rglob('manifest.csv'):
        if DEST in path.parents:
            continue
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                digest = row.get('sha256', '')
                if len(digest) == 64:
                    hashes.add(digest.lower())
    return hashes


def main():
    plan = ROOT / 'fresh_candidate_plan_v1.json'
    audit = json.loads((ROOT / 'fresh_candidate_member_audit.json').read_text())
    if audit['plan_sha256'] != hashlib.sha256(plan.read_bytes()).hexdigest():
        raise ValueError('Manifest changed after archive membership audit')
    rows = json.loads(plan.read_text())['samples']
    for row in rows:
        row['split'] = 'candidate'
    DEST.mkdir(exist_ok=False)
    result = []
    with ZipFile(original.REAL) as archive:
        for row in rows:
            if row['method'] == 'real':
                member = row['metadata_path'].removeprefix('deepfakes_detection_datasets/')
                data, digest = crop(archive, archive.getinfo(member))
                result.append(original.save(DEST, row, data, digest, member, 'local FF real archive'))
    original.SIZES.update(SIZES)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {method: pool.submit(original.fake_method, method,
                [row for row in rows if row['method'] == method], DEST) for method in SIZES}
        for job in jobs.values():
            result.extend(job.result())
    new_hashes = [row['sha256'] for row in result]
    overlap = sorted(set(new_hashes) & existing_hashes())
    report = {'plan_sha256': audit['plan_sha256'], 'total': len(result),
              'distinct_hashes': len(set(new_hashes)), 'prior_manifest_hash_overlaps': overlap,
              'limits': 'Hash checks do not establish identity or independence; no scores computed.'}
    with (DEST / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (DEST / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if len(result) != 210 or len(set(new_hashes)) != 210 or overlap:
        raise ValueError('Candidate image audit failed; preserve manifest for investigation')


if __name__ == '__main__':
    main()
