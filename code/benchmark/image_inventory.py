"""Read original Drive archives and the JPEG release without extracting images."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, JpegImagePlugin

GENERATORS = ['ADM', 'BigGAN', 'glide', 'Midjourney',
              'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
DCT = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32) + 0.5)) / 4
DCT[0] /= np.sqrt(2)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['empty'])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def reference_tables():
    data = io.BytesIO()
    Image.new('RGB', (8, 8)).save(data, format='JPEG', quality=96, subsampling=0)
    return Image.open(io.BytesIO(data.getvalue())).quantization


def properties(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        matched = (image.format == 'JPEG' and image.quantization == reference_tables()
                   and JpegImagePlugin.get_sampling(image) == 0)
        rgb = image.convert('RGB')
        small = np.asarray(rgb.convert('L').resize((32, 32)), dtype=float)
        low = (DCT @ small @ DCT.T).flatten()
        bits = low > np.median(low[1:])
        phash = sum(int(bit) << i for i, bit in enumerate(bits))
        pixels = str(rgb.size).encode() + rgb.tobytes()
        return {'pixel_sha256': digest(pixels), 'phash': f'{phash:016x}',
                'width': rgb.width, 'height': rgb.height, 'jpeg_match': int(matched)}


class Archives:
    def __init__(self, folder, release):
        self.original = [zipfile.ZipFile(p) for p in sorted(Path(folder).glob('GenImage_Compact*.zip'))]
        self.index = {}
        for archive in self.original:
            for name in archive.namelist():
                self.index[name.removeprefix('GenImage_Compact/')] = (archive, name)
        self.release = zipfile.ZipFile(release)

    def read(self, path):
        if path.startswith('_release_v2/'):
            return self.release.read(path)
        archive, name = self.index[path]
        return archive.read(name)

    def csv(self, path):
        return read_csv(self.read(path))


def scan(store, output):
    rows = []
    real = store.csv('_release_v2/audits/real_sha256.csv')
    maps = [r for g in GENERATORS for r in store.csv(f'_release_v2/conversion_maps/{g}.csv')]
    for i, item in enumerate(real + maps, 1):
        fake = 'v2_path' in item
        path = item['v2_path'] if fake else item['path']
        source = item['source'] if fake else path
        data = store.read(path)
        sha = digest(data)
        source_sha = digest(store.read(source)) if fake else sha
        if not fake and sha != item['sha256']:
            raise ValueError(f'Original real hash changed: {path}')
        row = {'path': path, 'label': int(fake), 'source_generator': item['generator'],
               'source_split': item['split'] if fake else item['source_split'],
               'sha256': sha, 'source_path': source, 'source_sha256': source_sha,
               'source_id': source_sha, **properties(data)}
        if fake and not row['jpeg_match']:
            raise ValueError(f'Unexpected converted JPEG settings: {path}')
        rows.append(row)
        if i % 1000 == 0:
            print(f'Inventory: {i}/32000 images decoded and hashed', flush=True)
    write_csv(output, rows)
    return rows
