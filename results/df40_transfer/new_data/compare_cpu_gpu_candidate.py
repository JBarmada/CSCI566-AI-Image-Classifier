"""Validate candidate prediction membership and compare CPU/P100 decisions."""
import csv
import json
import math
from pathlib import Path

from summarize_fresh_candidate import THRESHOLDS

ROOT = Path(__file__).resolve().parent
GPU = ROOT / 'fresh_candidate_results_v3'
CPU = ROOT / 'cpu_candidate_results_v1'
MANIFEST = ROOT / 'fresh_candidate_review_v3/manifest.csv'


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def keyed(path, expected):
    rows = read(path)
    result = {}
    for row in rows:
        key = (row['model'], row['mode'], row['sha256'])
        score = float(row['prob_fake'])
        if (key in result or key[:2] not in THRESHOLDS or row['sha256'] not in expected
                or (row['label'], row['method']) != expected[row['sha256']]
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f'Invalid or duplicate prediction: {key}')
        result[key] = score
    if len(result) != 945 or len(expected) != 189:
        raise ValueError('Unexpected sample or score count')
    return result


def main():
    acceptance = json.loads((GPU / 'ACCEPTANCE.json').read_text())
    if acceptance.get('status') != 'passed' or acceptance.get('scores') != 945:
        raise ValueError('GPU acceptance checks did not pass')
    expected = {r['sha256']: (r['label'], r['method']) for r in read(MANIFEST)}
    gpu, cpu = keyed(GPU / 'predictions.csv', expected), keyed(CPU / 'predictions.csv', expected)
    if set(gpu) != set(cpu):
        raise ValueError('CPU and GPU samples differ')
    output = []
    for model, mode in THRESHOLDS:
        pairs = [(gpu[k], cpu[k]) for k in gpu if k[:2] == (model, mode)]
        delta = [abs(a - b) for a, b in pairs]
        result = {'model': model, 'mode': mode, 'images': len(pairs),
                  'max_abs_difference': max(delta), 'mean_abs_difference': sum(delta) / len(delta)}
        for name, threshold in [('threshold_0_5', .5), ('previous_validation', THRESHOLDS[(model, mode)])]:
            result[name + '_flips'] = sum((a >= threshold) != (b >= threshold) for a, b in pairs)
        output.append(result)
    with (GPU / 'cpu_gpu_comparison.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
