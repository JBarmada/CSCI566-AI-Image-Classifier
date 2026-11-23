"""Retrieve 24 official frame crops, with no full archive downloads."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from remote_zip import RemoteZipFile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'smoke_v1'
METADATA = {
    'simswap': '1isStEussymA6ToZZHl87TqWJR0feobpA',
    'fomm': '1x7VF-A_1IWURHU6JZ60rIDaqWE14IvzO',
    'wav2lip': '1sYGDj4oX9MRffjtxXxflTIp2wjHsDFzt',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def candidates():
    path = ROOT / 'DF40_SMOKE_CANDIDATES.json'
    raw = path.read_bytes()
    data = json.loads(raw)
    provenance = {'selection_snapshot_sha256': digest(raw),
                  'selection': data['selection'], 'source_metadata': data['source_metadata']}
    return data['rows'], provenance


def extract_method(rows, method, archive):
    file_id = archive['url'].split('/d/')[1].split('/')[0]
    print('Reading ZIP directory:', method, flush=True)
    with zipfile.ZipFile(RemoteZipFile(file_id, archive['bytes'])) as source:
        names = source.namelist()
        for row in [r for r in rows if r['method'] == method]:
            suffix = '/'.join(row['official_path'].split('/')[-4:])
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f'Expected one member for {row["official_path"]}: {matches[:4]}')
            info = source.getinfo(matches[0])
            if info.file_size > 8 * 1024 * 1024:
                raise ValueError('Unexpectedly large image member')
            data = source.read(info)
            name = method + '_' + row['video_group'] + '_' + Path(matches[0]).name
            destination = OUT / 'images' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError('Existing smoke image changed')
            destination.write_bytes(data)
            with Image.open(destination) as image:
                image.load()
                row.update(width=image.width, height=image.height)
            row.update(path='images/' + name, sha256=digest(data), image_id=digest(data),
                       archive_url=archive['url'], archive_member=matches[0],
                       archive_crc32=f'{info.CRC:08x}', bytes=len(data))
            print('Retrieved', name, flush=True)


def extract(rows):
    with (ROOT / 'DF40_TEST_ARCHIVES.csv').open(encoding='utf-8') as stream:
        archives = {row['name'].removesuffix('.zip'): row for row in csv.DictReader(stream)}
    archives['real'] = {'url': 'https://drive.google.com/file/d/1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw/view',
                        'bytes': '2168805771'}
    errors = {}
    for method in [*METADATA, 'real']:
        try:
            extract_method(rows, method, archives[method])
        except Exception as error:
            errors[method] = str(error)
            print('Retrieval blocked:', method, error, flush=True)
    return errors


def save(rows, provenance, errors):
    rows = [r for r in rows if 'sha256' in r]
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate retrieved images')
    if not rows:
        report = {'status': 'blocked_no_images_retrieved', 'errors': errors,
                  'zip_range_bytes': RemoteZipFile.transferred, 'metadata': provenance}
        (OUT / 'RETRIEVAL.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    with (OUT / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in ['real', *METADATA]:
        selected = [r for r in rows if r['method'] == method]
        sheet = Image.new('RGB', (3 * 310, 2 * 340), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            with Image.open(OUT / row['path']) as source:
                thumbnail = ImageOps.contain(source.convert('RGB'), (300, 300))
            x, y = (i % 3) * 310, (i // 3) * 340
            sheet.paste(thumbnail, (x, y))
            draw.text((x, y + 302), f'{i+1}: {row["video_group"]}', fill='black')
        sheet.save(OUT / f'contact_{method}.jpg', quality=90)
    report = {'status': 'partial' if errors else 'retrieved_pending_visual_review_and_inference',
              'images': len(rows), 'errors': errors,
              'image_bytes': sum(r['bytes'] for r in rows),
              'zip_range_bytes': RemoteZipFile.transferred, 'metadata': provenance,
              'known_smoke_identities': sorted({i for r in rows for i in r['known_identity_ids'].split('|')}),
              'frame_edit_verification': 'Not established; official labels retained, no model-based filtering.',
              'exclusion_rule': 'Reserve these identities, video groups and related derivatives from final testing.'}
    (OUT / 'RETRIEVAL.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, provenance = candidates()
    errors = extract(rows)
    save(rows, provenance, errors)


if __name__ == '__main__':
    main()
