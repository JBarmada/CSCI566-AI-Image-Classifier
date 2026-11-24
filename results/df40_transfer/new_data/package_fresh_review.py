"""Package only retained candidate images for the team's restricted CARC area."""
import csv
import hashlib
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'fresh_candidate_data_v1'


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else 'v2'
    if version not in ('v2', 'v3'):
        raise ValueError('Expected v2 or v3')
    review = ROOT / f'fresh_candidate_review_{version}'
    archive_path = ROOT.parents[2] / 'tmp' / f'fresh_candidate_review_{version}.tar.gz'
    expected = 203 if version == 'v2' else 189
    with (review / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected or len({row['path'] for row in rows}) != expected:
        raise ValueError('Unexpected candidate manifest')
    with tarfile.open(archive_path, 'x:gz') as archive:
        archive.add(review / 'manifest.csv', arcname='manifest.csv')
        archive.add(review / 'audit.json', arcname='audit.json')
        for row in rows:
            source = DATA / row['path']
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Image hash mismatch: {source}')
            archive.add(source, arcname=row['path'])
    print(f'{archive_path} {archive_path.stat().st_size} bytes')
    print('sha256', hashlib.sha256(archive_path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
