"""Recheck archived ViT predictions without images, checkpoints, or sklearn."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'audit_sources/vit_benchmark'
BENCHMARK = ROOT / 'audit_sources/benchmark'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metrics(frame):
    y, p = frame.label.to_numpy(), frame.prob_fake.to_numpy()
    require(np.isin(y, [0, 1]).all(), 'Invalid labels')
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid scores')
    pred = p >= 0.5
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    tn, fn = int(((y == 0) & ~pred).sum()), int(((y == 1) & ~pred).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    require(pos > 0 and neg > 0, 'AUROC requires both labels')
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum()
              for b in range(10) if (bins == b).any()) / len(y)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {'accuracy': (tp + tn) / len(y), 'auroc': float(auc),
            'precision_fake': precision, 'recall_fake': recall,
            'f1_fake': 2 * precision * recall / max(precision + recall, 1e-30),
            'brier': float(np.mean((p - y) ** 2)), 'ece': float(ece),
            'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def audit_run(row, fingerprint):
    folder = SOURCE / row.holdout / 'runs' / f"{row.experiment}__{row['mode']}__seed17"
    config = load(folder / 'config.json')
    marker = load(folder / 'TRAINING_COMPLETE.json')
    require(not config['smoke_test'], 'Unexpected smoke checkpoint')
    require(config['dataset_fingerprint'] == fingerprint, 'Wrong release')
    require(config['protocol_hash'] == marker['protocol_hash'] == row.protocol_hash,
            'Training identity mismatch')
    evaluations = list(folder.glob('evaluations/clean_*/COMPLETE.json'))
    require(len(evaluations) == 1, 'Wrong evaluation count')
    evaluation = evaluations[0].parent
    result = load(evaluation / 'result.json')
    require(load(evaluations[0])['evaluation_hash'] == result['evaluation_hash']
            == row.evaluation_hash, 'Evaluation identity mismatch')
    count = 0
    for split in ['train', 'val', 'test']:
        path = BENCHMARK / 'manifests' / row.experiment / f'{split}.csv'
        require(hashlib.sha256(path.read_bytes()).hexdigest() == config['manifest_sha256'][split],
                f'Manifest hash mismatch: {path}')
        if split == 'train':
            continue
        expected = pd.read_csv(path).set_index('sha256').label
        directory = folder if split == 'val' else evaluation
        frame = pd.read_csv(directory / f'{split}_predictions.csv')
        require(len(frame) == 560 and frame.image_id.is_unique, 'Wrong prediction coverage')
        require(set(frame.image_id) == set(expected.index), 'Wrong images')
        require((frame.label == frame.image_id.map(expected)).all(), 'Wrong labels')
        require((frame.image_id == frame.sha256).all(), 'Identity columns disagree')
        require((frame.pred_label == (frame.prob_fake >= 0.5)).all(), 'Wrong threshold')
        computed = metrics(frame)
        saved = load(directory / f'{split}_metrics.json')
        require(all(abs(float(saved[k]) - v) <= 1e-6 for k, v in computed.items()),
                f'Saved metrics disagree: {directory}')
        if split == 'test':
            require(all(abs(float(row[k]) - v) <= 1e-6
                        and abs(float(result[k]) - v) <= 1e-6 for k, v in computed.items()),
                    'Summary metrics disagree')
        count += len(frame)
    return count


def main():
    fingerprint = load(BENCHMARK / 'RELEASE.json')['fingerprint']
    rows = pd.read_csv(SOURCE / 'summary/all_results.csv')
    plan = load(BENCHMARK / 'experiments.json')
    expected = {(e['experiment'], m) for e in plan for m in ['baseline', 'augmented']}
    require(len(rows) == 192 and set(zip(rows.experiment, rows['mode'])) == expected,
            'Incomplete run grid')
    require(set(rows.seed) == {17} and set(rows.condition) == {'clean'}, 'Wrong seed/condition')
    checked = 0
    for holdout, group in rows.groupby('holdout'):
        checked += sum(audit_run(row, fingerprint) for _, row in group.iterrows())
        print(f'Checked {holdout}: {len(group)} runs', flush=True)
    report = {'status': 'passed', 'runs': len(rows), 'prediction_files': len(rows) * 2,
              'prediction_rows': checked, 'dataset_fingerprint': fingerprint,
              'checks': ['manifest hashes', 'prediction identities and labels', 'threshold',
                         'training and evaluation markers', 'recomputed metrics including rank AUROC'],
              'not_rechecked': ['image bytes', 'checkpoint bytes', 'CNN raw predictions',
                                'paired CNN-ViT raw exports', 'robustness raw predictions']}
    destination = Path(__file__).parent / 'verification/VIT_CLEAN_REAUDIT.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
