"""Download the explicitly shared image subset and verify its manifest."""
import concurrent.futures
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'shared_dataset_v1'


def get_image(row):
    path = DATA / row['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = 'https://drive.google.com/uc?export=download&id=' + row['drive_id']
        with urllib.request.urlopen(url, timeout=45) as response:
            content = response.read(3_000_001)
        if len(content) > 3_000_000:
            raise ValueError('Unexpected image size: ' + row['path'])
        path.write_bytes(content)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Hash mismatch: ' + row['path'])
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        if image.size != (int(row['width']), int(row['height'])):
            raise ValueError('Dimension mismatch: ' + row['path'])
    return len(content)


def build_rows(index):
    rows = []
    for kind in ('real', 'fake'):
        for row in csv.DictReader(io.StringIO(index[kind + '_manifest'])):
            matches = [f for f in index['files'][row['batch']]
                       if f['name'] == Path(row['image_path']).name]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous file: ' + row['image_path'])
            row['path'] = kind + '/' + row['image_path']
            row['drive_id'] = matches[0]['id']
            rows.append(row)
    if len(rows) != 500 or len({r['sha256'] for r in rows}) != 500:
        raise ValueError('Unexpected sample count or duplicate hash')
    return rows


def main():
    index = json.loads((ROOT / 'SHARED_DRIVE_INDEX.json').read_text(encoding='utf-8'))
    rows = build_rows(index)
    DATA.mkdir(exist_ok=True)
    errors = []
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(get_image, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            try:
                total += future.result()
            except Exception as exc:
                errors.append({'path': futures[future]['path'], 'error': str(exc)})
    report = {'requested': len(rows), 'verified': len(rows) - len(errors),
              'bytes_verified': total, 'errors': errors}
    (DATA / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if errors:
        raise RuntimeError(json.dumps(report))
    with (DATA / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
