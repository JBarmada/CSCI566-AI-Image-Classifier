"""Audit public mirror ZIP indexes and selected samples with bounded reads."""
import concurrent.futures
import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from remote_zip import RemoteZipFile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'mirror_audit'
SOURCES = [
    ('ej2_simswap', 'simswap', 'ej2/df40',
     'e0d28c7b8ca85aa5a8102e1850e2c271fb23f24e', 2081311390),
    ('manh_simswap', 'simswap', 'ManhQuangAI/df-40-test-full',
     'b052e68b675c4c52b36439cfa1c88bdb49bb92b8', 2084511297),
    ('manh_fomm', 'fomm', 'ManhQuangAI/df-40-test-full',
     'b052e68b675c4c52b36439cfa1c88bdb49bb92b8', 1585909985),
    ('manh_wav2lip', 'wav2lip', 'ManhQuangAI/df-40-test-full',
     'b052e68b675c4c52b36439cfa1c88bdb49bb92b8', 2099533197),
]


def fingerprint(paths):
    a, b = 2166136261, 0x9e3779b9
    for byte in '\n'.join(sorted(set(paths))).encode('ascii'):
        a = ((a ^ byte) * 16777619) & 0xffffffff
        b = ((b ^ byte) * 2246822519) & 0xffffffff
    return f'{a:08x}{b:08x}'


def canonical_member(name, method):
    path = name.replace('\\', '/')
    marker = 'cdf/frames/'
    if marker not in path or not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return None
    return 'DF40/' + method + '/' + path[path.index(marker):]


def save_inventory(key, entries):
    path = OUT / f'{key}_cdf_inventory.csv.gz'
    with gzip.open(path, 'wt', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['official_path', 'archive_member', 'bytes', 'crc32'])
        writer.writerows((name, info.filename, info.file_size, f'{info.CRC:08x}')
                         for name, info in sorted(entries.items()))


def sample_images(source, entries, candidates, key):
    results = []
    directory = OUT / 'samples' / key
    directory.mkdir(parents=True, exist_ok=True)
    for row in candidates:
        info = entries.get(row['official_path'])
        if info is None:
            raise ValueError(f'Candidate missing: {row["official_path"]}')
        if info.file_size > 8 * 1024 * 1024 or info.flag_bits & 1:
            raise ValueError('Unexpected image size or encryption')
        payload = source.read(info)
        sha = hashlib.sha256(payload).hexdigest()
        path = directory / (row['video_group'] + '_' + Path(info.filename).name)
        if path.exists() and path.read_bytes() != payload:
            raise ValueError('Existing sample bytes changed')
        path.write_bytes(payload)
        with Image.open(path) as image:
            image.load()
            dimensions = [image.width, image.height]
        results.append({**row, 'archive_member': info.filename, 'bytes': len(payload),
                        'sha256': sha, 'dimensions': dimensions,
                        'path': path.relative_to(ROOT).as_posix(), 'crc_verified_by_zipfile': True})
        print('Verified sample:', key, path.name, flush=True)
    return results


def audit(spec, candidates, expected):
    key, method, repo, revision, size = spec
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{method}.zip?download=true'
    record = {'key': key, 'method': method, 'repository': repo, 'revision': revision,
              'archive_bytes': size, 'url': url, 'official_image_byte_identity_verified': False}
    try:
        print('Auditing archive:', key, flush=True)
        with zipfile.ZipFile(RemoteZipFile(None, size, url)) as source:
            entries = {}
            for info in source.infolist():
                canonical = canonical_member(info.filename, method)
                if canonical:
                    if canonical in entries:
                        raise ValueError('Duplicate canonical image path')
                    entries[canonical] = info
            groups = {name.split('/')[-2] for name in entries}
            expected_method = expected[method]['fake']
            metadata_ok = (len(entries) == expected_method['frames']
                           and len(groups) == expected_method['groups']
                           and fingerprint(entries) == expected_method['path_set_fingerprint'])
            record.update(cdf_images=len(entries), cdf_video_groups=len(groups),
                          metadata_path_set_matches=metadata_ok)
            save_inventory(key, entries)
            if not metadata_ok:
                raise ValueError('Mirror membership differs from official metadata')
            selected = [r for r in candidates if r['method'] == method]
            record['samples'] = sample_images(source, entries, selected, key)
            record['status'] = 'metadata_consistent_samples_readable'
    except Exception as error:
        record.update(status='failed', error=str(error))
    (OUT / f'{key}.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def compare_simswap():
    inventories = []
    for key in ['ej2_simswap', 'manh_simswap']:
        path = OUT / f'{key}_cdf_inventory.csv.gz'
        if not path.exists():
            return {'status': 'comparison_unavailable'}
        with gzip.open(path, 'rt', newline='', encoding='utf-8') as stream:
            inventories.append({r['official_path']: (r['bytes'], r['crc32']) for r in csv.DictReader(stream)})
    return {'status': 'compared', 'all_cdf_member_sizes_and_crc_match': inventories[0] == inventories[1],
            'compared_images': len(inventories[0]),
            'limit': 'CRC consistency is not a cryptographic verification against the official source.'}


def contact_sheet(records):
    selected = [r for rec in records if rec['key'] != 'ej2_simswap' for r in rec.get('samples', [])]
    if not selected:
        return
    sheet = Image.new('RGB', (6 * 270, 3 * 310), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        with Image.open(ROOT / row['path']) as image:
            thumb = ImageOps.contain(image.convert('RGB'), (260, 260))
        x, y = (i % 6) * 270, (i // 6) * 310
        sheet.paste(thumb, (x, y))
        draw.text((x, y + 263), row['method'] + ' ' + row['video_group'], fill='black')
    sheet.save(OUT / 'fake_candidates.jpg', quality=92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = json.loads((ROOT / 'DF40_SMOKE_CANDIDATES.json').read_text())['rows']
    expected = json.loads((ROOT / 'DF40_EXPECTED_METADATA.json').read_text())['expected']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda spec: audit(spec, candidates, expected), SOURCES))
    comparison = compare_simswap()
    sample_sets = [{r['official_path']: r['sha256'] for r in rec.get('samples', [])}
                   for rec in records if rec['method'] == 'simswap']
    comparison['selected_sample_sha256_match'] = (len(sample_sets) == 2
                                                 and bool(sample_sets[0]) and sample_sets[0] == sample_sets[1])
    contact_sheet(records)
    report = {'checked_date': '2026-11-21', 'sources': records,
              'simswap_cross_mirror_comparison': comparison,
              'range_bytes_downloaded': RemoteZipFile.transferred,
              'limits': ['Official image/checkpoint checksums not available for direct byte comparison.',
                         'Sample frame-edit success not established by dataset labels alone.',
                         'Real-image controls still required for complete smoke.']}
    (OUT / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'sources': [{k: r.get(k) for k in ['key', 'status', 'error', 'cdf_images',
          'cdf_video_groups', 'metadata_path_set_matches']} for r in records],
          'comparison': comparison, 'downloaded_bytes': RemoteZipFile.transferred}, indent=2))


if __name__ == '__main__':
    main()
