"""Audit and summarize CPU results without claiming GPU replay acceptance."""
import csv
import json
import math
from pathlib import Path

from summarize_fresh_candidate import METHODS, THRESHOLDS, summarize

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'cpu_candidate_results_v1'
MANIFEST = ROOT / 'fresh_candidate_review_v3/manifest.csv'


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    status = json.loads((FOLDER / 'STATUS.json').read_text())
    if status['status'] != 'exploratory_cpu_reference_mismatch' or status['scores'] != 945:
        raise ValueError('Unexpected CPU result status')
    expected = {row['sha256']: (row['label'], row['method']) for row in read(MANIFEST)}
    rows = read(FOLDER / 'predictions.csv')
    if len(expected) != 189 or len(rows) != 945:
        raise ValueError('Unexpected manifest or prediction count')
    for model, mode in THRESHOLDS:
        subset = [row for row in rows if row['model'] == model and row['mode'] == mode]
        if len(subset) != 189 or {row['sha256'] for row in subset} != set(expected):
            raise ValueError(f'Incomplete or duplicate predictions: {model} {mode}')
        for row in subset:
            score = float(row['prob_fake'])
            if (row['label'], row['method']) != expected[row['sha256']] or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid prediction, label or method')
    result = [summarize(rows, model, mode, method, threshold, name)
              for (model, mode), selected in THRESHOLDS.items() for method in METHODS
              for name, threshold in (('original_0.5', .5), ('previous_validation', selected))]
    with (FOLDER / 'metrics_preliminary.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f'Audited {len(rows)} preliminary CPU scores; wrote {len(result)} metric rows')


if __name__ == '__main__':
    main()
