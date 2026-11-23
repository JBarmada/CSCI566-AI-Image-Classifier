"""Retrieve DF40 FaceForensics++ fake crops from the fixed split plan."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from audit_df40_mirrors import SOURCES
from collect_fake_crops import RangeArchive
from remote_zip import RemoteZipFile
from verify_real_archive import crop

METHODS = ('simswap', 'fomm', 'wav2lip')
KEYS = {'ej2_simswap', 'manh_fomm', 'manh_wav2lip'}


def run_method(method, spec, records, dest):
    _, _, repo, rev, size = spec
    url = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{method}.zip?download=true'
    with ZipFile(RemoteZipFile(None, size, url)) as archive:
        entries = {}
        for item in archive.infolist():
            path = item.filename.replace('\\', '/')
            if 'ff/frames/' in path and path.endswith('.png'):
                key = path[path.index('ff/frames/'):]
                if key in entries:
                    raise ValueError(f'Duplicate archive entry: {key}')
                entries[key] = item
    source = RangeArchive(url, size)

    def retrieve(row):
        metadata = row['metadata_path']
        marker = 'ff/frames/'
        info = entries[metadata[metadata.index(marker):]]
        contents, digest = crop(source, info)
        relative = f'{row["split"]}/images/{method}/{row["group"]}.png'
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return {**row, 'path': relative, 'archive_member': info.filename,
                'archive_url': url, 'sha256': digest, 'width': 256, 'height': 256, 'label': 1}

    print(f'{method}: retrieving {len(records)} crops', flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(retrieve, records))


def main():
    plan = json.loads((ROOT / 'ff_split_plan.json').read_text())['samples']
    dest = ROOT / 'ff_fake_v1'
    dest.mkdir(exist_ok=False)
    sources = {s[1]: s for s in SOURCES if s[0] in KEYS}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {m: pool.submit(run_method, m, sources[m], [r for r in plan if r['method'] == m], dest)
                for m in METHODS}
        result = [row for m in METHODS for row in jobs[m].result()]
    if len(result) != 420 or len({r['sha256'] for r in result}) != 420:
        raise ValueError('Unexpected count or duplicate image bytes')
    with (dest / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f'Verified {len(result)} fake crops', flush=True)


if __name__ == '__main__':
    main()
