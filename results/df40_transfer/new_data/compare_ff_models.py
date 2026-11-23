"""Compare FaceForensics++ test scores and fixed validation thresholds."""
import csv
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def summarize(name, mode, rows, score_key, threshold):
    result = []
    for method in ('simswap', 'fomm', 'wav2lip'):
        real = [float(r[score_key]) for r in rows if r['method'] == 'real']
        fake = [float(r[score_key]) for r in rows if r['method'] == method]
        if len(real) != 70 or len(fake) != 70:
            raise ValueError('Unexpected test sample count')
        result.append({'model': name, 'mode': mode, 'method': method, 'threshold': threshold,
                       'real_fp': sum(p >= threshold for p in real), 'real_n': len(real),
                       'fake_tp': sum(p >= threshold for p in fake), 'fake_n': len(fake),
                       'auroc': roc_auc_score([0]*len(real) + [1]*len(fake), real + fake)})
    return result


def main():
    test = read(ROOT / 'ff_results_v1/test_predictions.csv')
    calibrated = json.loads((ROOT / 'ff_results_v1/test_report.json').read_text())['models']
    output = []
    for model in calibrated:
        selected = [r for r in test if r['model'] == model['model'] and r['mode'] == model['mode']]
        output += summarize(model['model'], model['mode'], selected, 'prob_fake', model['threshold'])
    zero = read(ROOT / 'ff_zeroshot_results/test_predictions.csv')
    threshold = json.loads((ROOT / 'ff_zeroshot_results/test_report.json').read_text())['models'][0]['threshold']
    output += summarize('clip_zero_shot', 'fixed_prompts', zero, 'prob_fake', threshold)
    adapted = read(ROOT / 'adaptation_results_v1/test_predictions.csv')
    threshold = json.loads((ROOT / 'adaptation_results_v1/RESULTS.json').read_text())['threshold']
    output += summarize('clip_adapted', 'supervised_head', adapted, 'prob_fake', threshold)
    with (ROOT / 'ff_model_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f'Compared {len(output) // 3} models across three manipulations')


if __name__ == '__main__':
    main()
