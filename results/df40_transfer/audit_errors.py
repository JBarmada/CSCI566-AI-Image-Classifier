"""Describe source subgroups and model errors without changing the dataset."""
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def main():
    rows = list(csv.DictReader((ROOT / 'shared_results_v1/predictions.csv').open()))
    results = []
    for model in ('cnn', 'vit'):
        for mode in ('baseline', 'augmented'):
            selected = [r for r in rows if r['model'] == model and r['mode'] == mode]
            for source in ('Celeb-real', 'YouTube-real'):
                group = [r for r in selected if r['method'] == 'real' and source in r['metadata_path']]
                false = sum(float(r['prob_fake']) >= .5 for r in group)
                results.append({'model': model, 'mode': mode, 'source': source,
                                'total': len(group), 'false_positives': false})
    output = ROOT / 'shared_results_v1/error_audit.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    real = [r for r in rows if r['method'] == 'real' and r['model'] == 'vit' and r['mode'] == 'baseline']
    cases = sorted(real, key=lambda r: float(r['prob_fake']), reverse=True)[:12]
    sheet = Image.new('RGB', (900, 340), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(cases):
        path = ROOT / 'shared_dataset_v1' / 'real' / row['image_path']
        with Image.open(path) as image:
            sheet.paste(image.convert('RGB').resize((140, 140)), ((i % 6) * 150, (i // 6) * 170))
        label = Path(row['image_path']).stem + ' p=' + format(float(row['prob_fake']), '.3f')
        draw.text(((i % 6) * 150, (i // 6) * 170 + 142), label, fill='black')
    sheet.save(ROOT / 'shared_results_v1/vit_real_false_positive_review.jpg')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
