"""Retrieve a small, reproducible DF40 CDF fake candidate pool."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from audit_df40_mirrors import SOURCES, canonical_member, fingerprint
from remote_zip import RemoteZipFile
from verify_real_archive import crop, read_manifest

METHODS = ('simswap', 'fomm', 'wav2lip')
SOURCE_KEYS = {'ej2_simswap', 'manh_fomm', 'manh_wav2lip'}
BASE = 'deepfakes_detection_datasets/'


def possible_clips(group):
    if re.fullmatch(r'(?:id\d+_)?\d{4,5}', group):
        return {group}
    swap = re.fullmatch(r'(id\d+)_(id\d+)_(\d{4})', group)
    if swap:
        first, second, number = swap.groups()
        return {f'{first}_{number}', f'{second}_{number}'}
    prefix = re.match(r'^(id\d+_\d{4})_', group)
    return {prefix[1]} if prefix else set()


def select_rows(method, expected, blocked, old_clips, count):
    payload = (ROOT / f'{method}_cdf.json').read_bytes()
    groups = json.loads(payload)[f'{method}_cdf'][f'{method}_Fake']['test']
    paths = {path.removeprefix(BASE) for entry in groups.values() for path in entry['frames']}
    if (len(groups), len(paths), fingerprint(paths)) != (
        expected['groups'], expected['frames'], expected['path_set_fingerprint']
    ):
        raise ValueError(f'{method} metadata differs from the official path inventory')
    ranked = sorted(set(groups) - blocked, key=lambda group: (
        len(possible_clips(group) & old_clips),
        hashlib.sha256(f'566:{method}:{group}'.encode()).digest(),
    ))
    selected = []
    for group in ranked[:count]:
        if not re.fullmatch(r'[\w-]+', group):
            raise ValueError(f'Unexpected group token: {group}')
        frames = sorted(groups[group]['frames'], key=lambda path: int(Path(path).stem))
        path = frames[(len(frames) - 1) // 2]
        selected.append((group, path))
    if len(selected) != count:
        raise ValueError(f'Not enough unused {method} groups')
    blocked.update(group for group, _ in selected)
    return selected, paths, hashlib.sha256(payload).hexdigest()


def archive_index(url, size, method):
    with ZipFile(RemoteZipFile(None, size, url)) as archive:
        entries = {}
        for info in archive.infolist():
            name = canonical_member(info.filename, method)
            if name is None:
                continue
            if name in entries:
                raise ValueError(f'Duplicate archive member: {name}')
            entries[name] = info
        return entries


def read_small_member(source, info):
    if info.file_size > 2_000_000 or info.compress_size > 2_000_000 or info.flag_bits & 1:
        raise ValueError(f'Unexpected archive member size or flags: {info.filename}')
    # Read the local header and data together, allowing for the ZIP extra field.
    source.seek(info.header_offset)
    window = source.read(30 + 2 * 65535 + info.compress_size)
    header = struct.unpack('<4s5H3I2H', window[:30])
    signature, _, flags, method = header[:4]
    name_length, extra_length = header[-2:]
    name = window[30:30 + name_length].decode('utf-8' if flags & 2048 else 'cp437')
    start = 30 + name_length + extra_length
    data = window[start:start + info.compress_size]
    if signature != b'PK\x03\x04' or name != info.filename or method != info.compress_type:
        raise ValueError(f'ZIP local header mismatch: {info.filename}')
    if len(data) != info.compress_size:
        raise ValueError(f'Truncated member: {info.filename}')
    if method == ZIP_DEFLATED:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        data = decoder.decompress(data, info.file_size + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError(f'Invalid deflate stream: {info.filename}')
    elif method != ZIP_STORED:
        raise ValueError(f'Unexpected ZIP compression: {method}')
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f'ZIP CRC or size mismatch: {info.filename}')
    return data


class RangeArchive:
    """Expose the same read interface used by the crop validator."""

    def __init__(self, url, size):
        self.url, self.size = url, size

    def read(self, info):
        for attempt in range(3):
            try:
                return read_small_member(RemoteZipFile(None, self.size, self.url), info)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)


def collect_method(method, spec, plan, old_clips, old_hashes, directory):
    _, _, repo, revision, size = spec
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{method}.zip?download=true'
    chosen, expected, metadata_hash = plan
    entries = archive_index(url, size, method)
    if not expected <= entries.keys():
        raise ValueError(f'{method} archive is missing official CDF paths')
    archive = RangeArchive(url, size)
    (directory / 'images' / method).mkdir(parents=True, exist_ok=True)

    def retrieve(item):
        group, path = item
        info = entries[path.removeprefix(BASE)]
        data, digest = crop(archive, info)
        if digest in old_hashes:
            raise ValueError(f'Fake crop duplicates earlier bytes: {group}')
        relative = f'images/{method}/{group}.png'
        (directory / relative).write_bytes(data)
        possible = possible_clips(group)
        return {'image_path': relative, 'path': relative, 'label': 1, 'method': method,
                'batch': 'candidate', 'video_group': group, 'metadata_path': path,
                'archive_member': info.filename, 'sha256': digest, 'width': 256, 'height': 256,
                'drive_id': '', 'archive_url': url, 'possible_source_clips': '|'.join(sorted(possible)),
                'possible_prior_clip_overlap': '|'.join(sorted(possible & old_clips))}

    print(f'{method}: {len(expected)} metadata paths cross-checked, retrieving {len(chosen)} crops', flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(retrieve, chosen))
    return rows, {'archive_url': url, 'metadata_sha256': metadata_hash,
                  'listed_paths_present': len(expected), 'mirror_cdf_paths': len(entries), 'samples': len(rows),
                  'possible_prior_clip_overlap': sum(bool(r['possible_prior_clip_overlap']) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=78)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 78:
        raise ValueError('Sample count must be between 1 and 78')
    with (ROOT / 'real_candidates_v1/manifest.csv').open(newline='', encoding='utf-8') as stream:
        real = list(csv.DictReader(stream))
    old = read_manifest()
    old_clips = set().union(*(possible_clips(row['video_group']) for row in old))
    blocked = {row['video_group'] for row in old + real}
    old_hashes = {row['sha256'] for row in old + real}
    expected = json.loads((ROOT.parent / 'DF40_EXPECTED_METADATA.json').read_text())['expected']
    sources = {spec[1]: spec for spec in SOURCES if spec[0] in SOURCE_KEYS}
    plan = {method: select_rows(method, expected[method]['fake'], blocked, old_clips, args.count)
            for method in METHODS}
    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {method: pool.submit(collect_method, method, sources[method], plan[method],
                                    old_clips, old_hashes, args.output) for method in METHODS}
        result = {method: job.result() for method, job in jobs.items()}
    rows = [row for method in METHODS for row in result[method][0]]
    if len({r['sha256'] for r in rows}) != len(rows) or len({r['video_group'] for r in rows}) != len(rows):
        raise ValueError('Duplicate crops or exact video-group names')
    with (args.output / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'counts': dict(Counter(r['method'] for r in rows)), 'transfer_bytes': RemoteZipFile.transferred,
              'methods': {method: result[method][1] for method in METHODS},
              'selection': 'Prioritize fewer inferred clip overlaps, then SHA-256 rank of seed 566/method/group; lower middle frame.',
              'limitations': 'Fake source clips are inferred from filenames. Exact group and byte exclusions '
                             'do not establish source or identity independence. Labels follow DF40 metadata.'}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
