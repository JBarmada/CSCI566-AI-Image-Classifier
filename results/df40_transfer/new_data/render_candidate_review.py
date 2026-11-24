"""Render the first three retained groups for manual crop review."""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'fresh_candidate_review_v2' / 'manifest.csv'
DATA = ROOT / 'fresh_candidate_data_v1'
OUTPUT = ROOT.parents[2] / 'tmp' / 'fresh_face_review.png'
METHODS = ('real', 'real', 'simswap', 'fomm', 'inswap', 'pirender', 'wav2lip')


def main():
    with MANIFEST.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    pairs = list(dict.fromkeys(row['pair'] for row in rows))[:3]
    canvas = Image.new('RGB', (7 * 180, 3 * 210), 'white')
    draw = ImageDraw.Draw(canvas)
    for line, pair in enumerate(pairs):
        samples = [row for row in rows if row['pair'] == pair]
        for col, method in enumerate(METHODS):
            candidates = [row for row in samples if row['method'] == method]
            row = candidates[1 if method == 'real' and col == 1 else 0]
            with Image.open(DATA / row['path']) as image:
                canvas.paste(image.convert('RGB').resize((170, 170)), (col * 180, line * 210 + 30))
            draw.text((col * 180 + 2, line * 210 + 3), f'{pair}  {method}', fill='black')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()
