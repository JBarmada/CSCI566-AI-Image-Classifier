"""Retrieve the predeclared small FF training and validation sample."""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from collect_fake_crops import RangeArchive
from remote_zip import RemoteZipFile
from verify_real_archive import crop

REV = 'ef0577c67506afea6df7a8bb46dd4ed4a8e138a0'
SIZES = {'simswap': 2922323911, 'fomm': 1635613346, 'wav2lip': 2112480168}
REAL = Path(r'C:\Users\creat\Desktop\school\CSCI566\AI project\FaceForensics++_real_data_for_DF40.zip')


def save(dest, row, contents, digest, member, url):
    relative = f'{row["split"]}/images/{row["method"]}/{row["group"]}.png'
    path = dest / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return {'split': row['split'], 'method': row['method'], 'video_group': row['group'],
            'source_ids': '|'.join(row['source_ids']), 'path': relative, 'sha256': digest,
            'label': int(row['method'] != 'real'), 'metadata_path': row['metadata_path'],
            'archive_member': member, 'archive_url': url}


def fake_method(method, plan, dest):
    url = f'https://huggingface.co/datasets/ManhQuangAI/DF40_train/resolve/{REV}/{method}.zip?download=true'
    with ZipFile(RemoteZipFile(None, SIZES[method], url)) as archive:
        entries = {item.filename: item for item in archive.infolist()}
    source = RangeArchive(url, SIZES[method])

    def fetch(row):
        path = row['metadata_path']
        member = path[path.index(method + '/frames/'):]
        data, digest = crop(source, entries[member])
        return save(dest, row, data, digest, member, url)

    with ThreadPoolExecutor(max_workers=6) as pool:
        result = list(pool.map(fetch, plan))
    print(method, len(result), 'verified', flush=True)
    return result


def main():
    plan_file = ROOT / 'adaptation_plan.json'
    plan = json.loads(plan_file.read_text())['samples']
    dest = ROOT / 'adaptation_data_v1'
    dest.mkdir(exist_ok=False)
    result = []
    with ZipFile(REAL) as archive:
        for row in plan:
            if row['method'] != 'real':
                continue
            member = row['metadata_path'].removeprefix('deepfakes_detection_datasets/')
            data, digest = crop(archive, archive.getinfo(member))
            result.append(save(dest, row, data, digest, member, 'local official FF real archive'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(fake_method, m, [r for r in plan if r['method'] == m], dest) for m in SIZES]
        for job in jobs:
            result.extend(job.result())
    if len(result) != 360 or len({r['sha256'] for r in result}) != len(result):
        raise ValueError('Unexpected count or duplicate bytes')
    with (dest / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    report = {'samples': len(result), 'plan_sha256': hashlib.sha256(plan_file.read_bytes()).hexdigest(),
              'validation': 'Source ID groups separated by fixed plan; identity independence unknown.'}
    (dest / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
