"""Compare two mirror ZIP indexes with separately published DF40 metadata."""
import json
from pathlib import Path
import sys
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from remote_zip import RemoteZipFile

REV = 'ef0577c67506afea6df7a8bb46dd4ed4a8e138a0'
META_REV = 'a203e9b880d69339178f1a6aacd937b07a8ef199'
SIZES = {'inswap': 2502440755, 'pirender': 1763272422}
META = f'https://huggingface.co/datasets/sinister007/df40_dataset/resolve/{META_REV}/dataset_json/'


def inspect(method, size):
    url = f'https://huggingface.co/datasets/ManhQuangAI/DF40_train/resolve/{REV}/{method}.zip?download=true'
    with urllib.request.urlopen(META + method + '_ff.json', timeout=60) as response:
        metadata = json.load(response)[method + '_ff']
    groups = metadata[method + '_Fake']['train']
    expected = set()
    marker = f'{method}/frames/'
    for item in groups.values():
        for path in item['frames']:
            if marker not in path:
                raise ValueError(f'Unexpected metadata path: {path}')
            expected.add(method + '/frames/' + path.split(marker, 1)[1])
    with ZipFile(RemoteZipFile(None, size, url)) as archive:
        actual = {item.filename.replace('\\', '/') for item in archive.infolist()
                  if not item.is_dir() and item.filename.lower().endswith('.png')}
    return {'archive_url': url, 'archive_bytes': size, 'metadata_url': META + method + '_ff.json',
            'metadata_train_frames': len(expected), 'archive_png_frames': len(actual),
            'metadata_frames_found_in_archive': len(expected & actual),
            'metadata_frames_missing_from_archive': len(expected - actual),
            'extra_archive_png_frames': len(actual - expected),
            'metadata_groups': len(groups), 'bytes_identical_to_official': None}


def main():
    results = {method: inspect(method, size) for method, size in SIZES.items()}
    (ROOT / 'extra_method_archive_audit.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
