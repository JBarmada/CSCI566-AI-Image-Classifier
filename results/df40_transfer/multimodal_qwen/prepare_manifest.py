"""Freeze a small, class-balanced shared GenImage comparison subset."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'clean_cnn_audit/prediction_export/runs/holdout_ADM_k7_rep0__baseline__seed17/test_predictions.csv'


def rank(row):
    return hashlib.sha256(('566:multimodal:' + row['sha256']).encode()).digest()


def main():
    with SOURCE.open(newline='', encoding='utf-8') as stream:
        original = list(csv.DictReader(stream))
    rows = []
    for label in ('0', '1'):
        chosen = sorted((row for row in original if row['label'] == label), key=rank)[:32]
        rows.extend({'path': r['path'], 'sha256': r['sha256'], 'label': r['label'],
                     'source': r['generator']} for r in chosen)
    if len(rows) != 64 or len({r['sha256'] for r in rows}) != 64:
        raise ValueError('Unexpected selected samples')
    rows.sort(key=rank)
    with (ROOT / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {'model': 'Qwen/Qwen2.5-VL-3B-Instruct',
                'revision': '66285546d2b821cf421d4f5eb2576359d3770cd3',
                'prompt': 'Is this image a real photograph or an AI-generated image? '
                          'Choose the more likely option. Reply with exactly REAL or FAKE.',
                'generation': {'do_sample': False, 'max_new_tokens': 8},
                'subset': '32 real and 32 ADM fake; SHA-256 rank with seed 566; shared ADM k7 rep0 test images',
                'selection': 'Frozen before model inference. Labels are not included in model input.',
                'interpretation': 'Greedy text labels only; invalid responses retained and reported.'}
    with (ROOT / 'protocol.json').open('x', encoding='utf-8') as stream:
        json.dump(settings, stream, indent=2)
        stream.write('\n')
    print('Frozen 64-image manifest and prompt')


if __name__ == '__main__':
    main()
