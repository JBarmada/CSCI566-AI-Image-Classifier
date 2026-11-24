"""Package only retained candidate images for the team's restricted CARC area."""
import csv
import hashlib
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / 'fresh_candidate_review_v2'
DATA = ROOT / 'fresh_candidate_data_v1'
ARCHIVE = ROOT.parents[2] / 'tmp' / 'fresh_candidate_review_v2.tar.gz'


def main():
    with (REVIEW / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 203 or len({row['path'] for row in rows}) != 203:
        raise ValueError('Unexpected candidate manifest')
    with tarfile.open(ARCHIVE, 'x:gz') as archive:
        archive.add(REVIEW / 'manifest.csv', arcname='manifest.csv')
        archive.add(REVIEW / 'audit.json', arcname='audit.json')
        for row in rows:
            source = DATA / row['path']
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Image hash mismatch: {source}')
            archive.add(source, arcname=row['path'])
    print(f'{ARCHIVE} {ARCHIVE.stat().st_size} bytes')
    print('sha256', hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
