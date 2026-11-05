"""Audit all 192 exported evaluations, then produce cross-holdout summaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def require(ok, message):
    if not ok:
        raise ValueError(message)


def independent_metrics(frame):
    y, p = frame.label.to_numpy(), frame.prob_fake.to_numpy()
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
    pred = p >= .5
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    tn, fn = int(((y == 0) & ~pred).sum()), int(((y == 1) & ~pred).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == i].mean() - y[bins == i].mean()) * (bins == i).sum()
              for i in range(10) if (bins == i).any()) / len(y)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {'accuracy': (tp + tn) / len(y), 'precision_fake': precision, 'recall_fake': recall,
            'f1_fake': 2 * precision * recall / max(precision + recall, 1e-30),
            'auroc': roc_auc_score(y, p), 'brier': np.mean((p - y) ** 2), 'ece': ece,
            'log_loss': -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def check_run(root, benchmark, experiment, mode, fingerprint):
    name = experiment['experiment']
    folder = root / experiment['holdout'] / 'runs' / f'{name}__{mode}__seed17'
    config = json.loads((folder / 'config.json').read_text())
    require(not config['smoke_test'] and config['dataset_fingerprint'] == fingerprint, 'Wrong dataset/study')
    marker = json.loads((folder / 'TRAINING_COMPLETE.json').read_text())
    require(marker['protocol_hash'] == config['protocol_hash'], 'Training protocol mismatch')
    require(marker['checkpoint_sha256'] == hashlib.sha256((folder / 'best_head.pt').read_bytes()).hexdigest(),
            'Checkpoint hash mismatch')
    evaluations = list(folder.glob('evaluations/clean_*/COMPLETE.json'))
    require(len(evaluations) == 1, f'{name}/{mode}: expected one completed clean evaluation')
    evaluation = evaluations[0].parent
    result = json.loads((evaluation / 'result.json').read_text())
    complete = json.loads(evaluations[0].read_text())
    require(complete['evaluation_hash'] == result['evaluation_hash'], 'Evaluation marker mismatch')
    require(result['protocol_hash'] == config['protocol_hash'], 'Evaluation protocol mismatch')
    for split, destination, stem in [('val', folder, 'val'), ('test', evaluation, 'test')]:
        csv = benchmark / 'manifests' / name / f'{split}.csv'
        require(hashlib.sha256(csv.read_bytes()).hexdigest() == config['manifest_sha256'][split], 'CSV changed')
        expected = pd.read_csv(csv)
        frame = pd.read_csv(destination / f'{stem}_predictions.csv')
        require(len(frame) == 560 and frame.image_id.is_unique, 'Prediction rows missing/duplicated')
        require(set(frame.image_id) == set(expected.sha256), 'Wrong prediction image identities')
        labels = expected.set_index('sha256').label
        require((frame.label.to_numpy() == frame.image_id.map(labels).to_numpy()).all(), 'Labels changed')
        require((frame.pred_label == (frame.prob_fake >= .5).astype(int)).all(), 'Threshold mismatch')
        computed = independent_metrics(frame)
        saved = json.loads((destination / f'{stem}_metrics.json').read_text())
        require(all(abs(saved[k] - float(v)) <= 1e-6 for k, v in computed.items()), 'Metrics disagree')
        if split == 'test':
            require(all(abs(result[k] - float(v)) <= 1e-6 for k, v in computed.items()), 'Summary metrics disagree')
    return result


def finalize(shared):
    benchmark = shared / 'data/extracted/GenImage_Compact/benchmark'
    fingerprint = json.loads((benchmark / 'RELEASE.json').read_text())['fingerprint']
    experiments = json.loads((benchmark / 'experiments.json').read_text())
    root = shared / 'results/vit_benchmark'
    results = [check_run(root, benchmark, exp, mode, fingerprint)
               for exp in experiments for mode in ['baseline', 'augmented']]
    require(len(results) == 192, 'Expected 192 evaluations')
    frame = pd.DataFrame(results)
    output = root / 'summary'
    output.mkdir(exist_ok=True)
    frame.to_csv(output / 'all_results.csv', index=False)
    metrics = ['accuracy', 'f1_fake', 'auroc', 'ece', 'brier', 'log_loss']
    table = frame.groupby(['holdout', 'k', 'mode'])[metrics].agg(['mean', 'std', 'count'])
    table.columns = ['_'.join(c) for c in table.columns]
    table.reset_index().to_csv(output / 'by_holdout_k.csv', index=False)
    # Equal-weight holdouts; do not treat repeated predictions on fixed tests as independent images.
    holdout_means = frame.groupby(['holdout', 'k', 'mode'])[metrics].mean().reset_index()
    macro = holdout_means.groupby(['k', 'mode'])[metrics].agg(['mean', 'std', 'count'])
    macro.columns = ['_'.join(c) for c in macro.columns]
    macro.reset_index().to_csv(output / 'macro_across_holdouts.csv', index=False)
    report = {'status': 'passed', 'completed_evaluations': 192, 'prediction_exports_checked': 384,
              'dataset_fingerprint': fingerprint, 'independently_recomputed_metrics': True}
    (output / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path, default=Path('/project2/yzhao010_1531/csci566-genimage'))
    finalize(parser.parse_args().shared_root)
