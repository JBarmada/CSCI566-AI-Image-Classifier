"""Create a contact sheet from filename-linked source candidates."""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    prior = rows(ROOT.parent / 'shared_dataset_v1/manifest.csv')
    real = {r['video_group']: r for r in prior if r['method'] == 'real'}
    fake = rows(ROOT / 'fake_candidates_v1/manifest.csv')
    pairs, used = [], set()
    for method in ('simswap', 'fomm', 'wav2lip'):
        options = [r for r in fake if r['method'] == method]
        for row in options:
            matches = [x for x in row['possible_prior_clip_overlap'].split('|') if x in real and x not in used]
            if matches:
                group = matches[0]
                pairs.append((real[group], row))
                used.add(group)
                if sum(p[1]['method'] == method for p in pairs) == 2:
                    break
    if not pairs:
        raise ValueError('No filename-linked real/fake pairs found')
    canvas = Image.new('RGB', (660, len(pairs) * 245), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (old, new) in enumerate(pairs):
        y = i * 245
        a = ROOT.parent / 'shared_dataset_v1' / old['path']
        b = ROOT / 'fake_candidates_v1' / new['path']
        for path, x in ((a, 10), (b, 330)):
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB').resize((205, 205)), (x, y + 25))
        draw.text((10, y + 3), f'{i+1}  Earlier real: {old["video_group"]}', fill='black')
        draw.text((330, y + 3), f'{new["method"]}: {new["video_group"]}', fill='black')
    canvas.save(ROOT / 'possible_source_pairs.jpg', quality=92)
    print('Created', ROOT / 'possible_source_pairs.jpg')


if __name__ == '__main__':
    main()
