"""Create six-pair contact sheets for manual candidate crop inspection."""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'fresh_candidate_review_v2/manifest.csv'
DATA = ROOT / 'fresh_candidate_data_v1'
OUTPUT = ROOT.parents[2] / 'tmp/candidate_contact_sheets'
METHODS = ('real', 'real', 'simswap', 'fomm', 'inswap', 'pirender', 'wav2lip')


def render(rows, pairs, page):
    canvas = Image.new('RGB', (7 * 150, len(pairs) * 178), 'white')
    draw = ImageDraw.Draw(canvas)
    for line, pair in enumerate(pairs):
        samples = [row for row in rows if row['pair'] == pair]
        for col, method in enumerate(METHODS):
            candidates = [row for row in samples if row['method'] == method]
            row = candidates[1 if method == 'real' and col == 1 else 0]
            with Image.open(DATA / row['path']) as image:
                canvas.paste(image.convert('RGB').resize((140, 140)), (col * 150, line * 178 + 25))
            draw.text((col * 150 + 2, line * 178 + 3), f'{pair} {method}', fill='black')
    canvas.save(OUTPUT / f'page_{page}.png')


def main():
    with MANIFEST.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    pairs = list(dict.fromkeys(row['pair'] for row in rows))
    if len(pairs) != 29:
        raise ValueError('Unexpected candidate pair count')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), 6):
        render(rows, pairs[start:start + 6], start // 6 + 1)
    print(f'Rendered {len(pairs)} pairs to {OUTPUT}')


if __name__ == '__main__':
    main()
