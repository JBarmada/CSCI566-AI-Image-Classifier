"""Verify the local DF40 real ZIP and extract one crop per unused listed test group."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from audit_candidates import ids

ROOT = Path(__file__).resolve().parent
PREFIX = 'deepfakes_detection_datasets/'


def read_manifest():
    path = ROOT.parent / 'shared_dataset_v1/manifest.csv'
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def fingerprint(paths):
    first, second = 2166136261, 0x9e3779b9
    for byte in '\n'.join(sorted(set(paths))).encode('ascii'):
        first = ((first ^ byte) * 16777619) & 0xffffffff
        second = ((second ^ byte) * 2246822519) & 0xffffffff
    return f'{first:08x}{second:08x}'


def listed_groups():
    payload = (ROOT / 'simswap_cdf.json').read_bytes()
    groups = json.loads(payload)['simswap_cdf']['simswap_Real']['test']
    expected = json.loads((ROOT.parent / 'DF40_EXPECTED_METADATA.json').read_text())
    reference = expected['expected']['simswap']['real']
    paths = [path.removeprefix(PREFIX) for group in groups.values() for path in group['frames']]
    if len(groups) != reference['groups'] or len(set(paths)) != reference['frames']:
        raise ValueError('The real metadata counts disagree with the archived official metadata')
    if fingerprint(paths) != reference['path_set_fingerprint']:
        raise ValueError('The real path list disagrees with the archived official metadata')
    return groups, paths, hashlib.sha256(payload).hexdigest()


def crop(archive, info):
    data = archive.read(info)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != 'PNG' or image.size != (256, 256):
            raise ValueError(f'Unexpected crop format or size: {info.filename}')
    return data, hashlib.sha256(data).hexdigest()


def select(archive, members, groups, old):
    old_groups = {row['video_group'] for row in old}
    old_ids = set().union(*(ids(group) for group in old_groups))
    old_hashes = {row['sha256'] for row in old}
    old_real = [row for row in old if row['method'] == 'real']
    for row in old_real:
        _, digest = crop(archive, members[row['archive_member']])
        if digest != row['sha256']:
            raise ValueError(f'Previously evaluated real crop differs: {row["archive_member"]}')
    candidates = []
    for group, record in sorted(groups.items()):
        if group in old_groups:
            continue
        frames = sorted(record['frames'], key=lambda path: int(Path(path).stem))
        path = frames[(len(frames) - 1) // 2]
        member = path.removeprefix(PREFIX)
        data, digest = crop(archive, members[member])
        if digest in old_hashes:
            raise ValueError(f'Duplicate crop bytes in candidate pool: {member}')
        old_hashes.add(digest)
        tokens = ids(group)
        overlap = 'known-token-overlap' if tokens & old_ids else 'unknown' if not tokens else 'new-token'
        relative = f'images/{group}.png'
        candidates.append(({
            'image_path': relative, 'path': relative, 'label': 0, 'method': 'real',
            'batch': 'candidate', 'video_group': group, 'metadata_path': path,
            'archive_member': member, 'sha256': digest, 'width': 256, 'height': 256,
            'drive_id': '', 'identity_status': overlap,
        }, data))
    return candidates, len(old_real)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups, paths, meta_hash = listed_groups()
    with ZipFile(args.archive) as archive:
        images = [item for item in archive.infolist() if item.filename.endswith('.png')]
        members = {item.filename: item for item in images}
        if len(images) != len(members) or not set(paths) <= members.keys():
            raise ValueError('Duplicate archive paths or missing metadata-listed crops')
        candidates, verified = select(archive, members, groups, read_manifest())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    for row, data in candidates:
        (output / row['path']).write_bytes(data)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0][0]))
        writer.writeheader()
        writer.writerows(row for row, _ in candidates)
    report = {
        'archive_filename': args.archive.name, 'archive_size': args.archive.stat().st_size,
        'archive_png_members': len(images), 'metadata_sha256': meta_hash,
        'metadata_video_groups': len(groups), 'metadata_listed_crops_present': len(set(paths)),
        'previously_evaluated_real_sha256_verified': verified, 'candidate_groups': len(candidates),
        'identity_status': dict(Counter(row['identity_status'] for row, _ in candidates)),
        'source': dict(Counter(row['archive_member'].split('/')[1] for row, _ in candidates)),
        'selection': 'One lower-middle frame by numeric frame ID per unused exact metadata group',
        'limitations': 'Candidate pool only. Exact group names and bytes were checked. '
                       'Related source clips and unknown identities have not been resolved.',
    }
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
