"""Extract planned FaceForensics++ real crops and combine split manifests."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from verify_real_archive import crop

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(r'C:\Users\creat\Desktop\school\CSCI566\AI project\FaceForensics++_real_data_for_DF40.zip')


def main():
    plan = json.loads((ROOT / 'ff_split_plan.json').read_text())['samples']
    fake_root = ROOT / 'ff_fake_v1'
    with (fake_root / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        fake = list(csv.DictReader(stream))
    dest = ROOT / 'ff_complete_v1'
    dest.mkdir(exist_ok=False)
    output = []
    with ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        for row in plan:
            if row['method'] != 'real':
                continue
            member = row['metadata_path'].removeprefix('deepfakes_detection_datasets/')
            if member not in names:
                raise ValueError(f'Missing original crop {member}')
            contents, digest = crop(archive, archive.getinfo(member))
            relative = f'images/real/{row["group"]}.png'
            target = dest / row['split'] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            output.append({'split': row['split'], 'path': relative, 'method': 'real',
                           'video_group': row['group'], 'sha256': digest, 'label': '0',
                           'metadata_path': row['metadata_path'], 'archive_member': member})
    for row in fake:
        src = fake_root / row['path']
        parts = Path(row['path']).parts
        relative = Path(*parts[1:]).as_posix()
        target = dest / row['split'] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        output.append({'split': row['split'], 'path': relative, 'method': row['method'],
                       'video_group': row['group'], 'sha256': row['sha256'], 'label': '1',
                       'metadata_path': row['metadata_path'], 'archive_member': row['archive_member']})
    if len(output) != 560 or len({r['sha256'] for r in output}) != 560:
        raise ValueError('Unexpected count or duplicated image')
    for split in ('validation', 'test'):
        selected = [row for row in output if row['split'] == split]
        with (dest / split / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    print('Prepared 280 images per split')


if __name__ == '__main__':
    main()
