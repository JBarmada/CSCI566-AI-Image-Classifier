"""Audit downloaded clean predictions and publish analysis tables without training."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from finalize_benchmark import independent_metrics, require

METRICS = ['accuracy', 'precision_fake', 'recall_fake', 'f1_fake', 'auroc',
           'ece', 'brier', 'log_loss']
KEYS = ['experiment', 'holdout', 'k', 'rep', 'seed', 'mode']


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def csv_gzip(frame, path):
    frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})


def verify_prediction(path, manifest, metrics_path):
    frame, expected = pd.read_csv(path), pd.read_csv(manifest)
    require(len(frame) == 560 and frame.image_id.is_unique, f'Invalid rows: {path}')
    require(set(frame.image_id) == set(expected.sha256), f'Image mismatch: {path}')
    labels = frame.image_id.map(expected.set_index('sha256').label)
    require((frame.label == labels).all(), f'Label mismatch: {path}')
    require((frame.sha256 == frame.image_id).all(), f'Identity mismatch: {path}')
    require((frame.pred_label == (frame.prob_fake >= .5)).all(), f'Threshold: {path}')
    computed, saved = independent_metrics(frame), read_json(metrics_path)
    for key, value in computed.items():
        require(abs(float(saved[key]) - value) <= 1e-6, f'Metric {key}: {path}')
    return frame, computed


def process_run(source, benchmark, row):
    folder = source / row.holdout / 'runs' / f"{row.experiment}__{row['mode']}__seed17"
    config, marker = read_json(folder / 'config.json'), read_json(folder / 'TRAINING_COMPLETE.json')
    evaluation = list(folder.glob('evaluations/clean_*/COMPLETE.json'))
    require(len(evaluation) == 1, f'Invalid evaluation count: {folder}')
    evaluation = evaluation[0].parent
    result = read_json(evaluation / 'result.json')
    require(config['protocol_hash'] == marker['protocol_hash'] == row.protocol_hash,
            f'Protocol mismatch: {folder}')
    require(result['evaluation_hash'] == row.evaluation_hash, 'Evaluation mismatch')
    for split in ['train', 'val', 'test']:
        manifest = benchmark / 'manifests' / row.experiment / f'{split}.csv'
        require(hashlib.sha256(manifest.read_bytes()).hexdigest() == config['manifest_sha256'][split],
                f'Manifest hash mismatch: {manifest}')
    predictions, calibrations, val_metrics = [], [], None
    for split, directory in [('val', folder), ('test', evaluation)]:
        manifest = benchmark / 'manifests' / row.experiment / f'{split}.csv'
        prediction, computed = verify_prediction(directory / f'{split}_predictions.csv',
                                                  manifest, directory / f'{split}_metrics.json')
        if split == 'test':
            require(all(abs(float(row[k]) - v) <= 1e-6 for k, v in computed.items()), 'Summary mismatch')
        else:
            val_metrics = computed
        prediction['split'], prediction['holdout'] = split, row.holdout
        prediction['k'], prediction['rep'] = row.k, row.rep
        predictions.append(prediction)
        calibration = pd.read_csv(directory / f'{split}_calibration.csv')
        require(len(calibration) == 10 and calibration.bin.is_unique, 'Invalid calibration bins')
        bin_index = np.minimum((prediction.prob_fake.to_numpy() * 10).astype(int), 9)
        for _, bin_row in calibration.iterrows():
            selected = prediction[bin_index == int(bin_row['bin'])]
            require(len(selected) == bin_row['count'], 'Calibration bin count mismatch')
            if len(selected):
                require(abs(selected.prob_fake.mean() - bin_row.mean_prob_fake) <= 1e-6
                        and abs(selected.label.mean() - bin_row.observed_fake_fraction) <= 1e-6,
                        'Calibration bin value mismatch')
        for key in KEYS:
            calibration[key] = row[key]
        calibration['split'] = split
        calibrations.append(calibration)
    history = pd.read_csv(folder / 'training_history.csv')
    require(history.epoch.is_unique and marker['best_epoch'] in set(history.epoch), 'Invalid epochs')
    for key in KEYS:
        history[key] = row[key]
    detail = dict(row)
    detail.update({'val_' + k: v for k, v in val_metrics.items()})
    detail.update(best_epoch=marker['best_epoch'], epochs_recorded=len(history),
                  epoch_seconds_recorded=history.seconds.sum(), completed_utc=marker['completed_utc'])
    for key, value in config.items():
        detail['config_' + key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    raw = {'config': config, 'training_complete': marker, 'result': result,
           'evaluation_complete': read_json(evaluation / 'COMPLETE.json')}
    return detail, predictions, calibrations, history, raw


def aggregated(frame, keys):
    result = frame.groupby(keys)[METRICS].agg(['mean', 'std', 'count'])
    result.columns = ['_'.join(c) for c in result.columns]
    return result.reset_index()


def derived_tables(details, output):
    aggregated(details, ['holdout', 'k', 'mode']).to_csv(output / 'by_holdout_k.csv', index=False)
    holdout_means = details.groupby(['holdout', 'k', 'mode'])[METRICS].mean().reset_index()
    aggregated(holdout_means, ['k', 'mode']).to_csv(output / 'macro_across_holdouts.csv', index=False)
    baseline = details[details['mode'] == 'baseline'].set_index(KEYS[:-1])
    augmented = details[details['mode'] == 'augmented'].set_index(KEYS[:-1])
    delta = (augmented[METRICS] - baseline[METRICS]).add_suffix('_aug_minus_base').reset_index()
    delta.to_csv(output / 'augmentation_deltas.csv', index=False)
    gap = details[KEYS].copy()
    for metric in METRICS:
        gap[metric + '_val_minus_test'] = details['val_' + metric] - details[metric]
    gap.to_csv(output / 'validation_test_gaps.csv', index=False)
    for metric in ['accuracy', 'auroc', 'f1_fake', 'recall_fake']:
        holdout_means.pivot(index='holdout', columns=['mode', 'k'], values=metric).to_csv(
            output / f'heatmap_{metric}.csv')


def export(source, benchmark, output):
    output.mkdir(parents=True, exist_ok=True)
    for directory in ['summary', 'metadata', 'predictions']:
        (output / directory).mkdir(exist_ok=True)
    acceptance = read_json(source / 'summary/ACCEPTANCE.json')
    release = read_json(benchmark / 'RELEASE.json')
    require(acceptance['status'] == 'passed' and acceptance['dataset_fingerprint'] == release['fingerprint'],
            'Wrong release or failed source audit')
    results = pd.read_csv(source / 'summary/all_results.csv')
    require(len(results) == 192 and not results.duplicated(KEYS).any(), 'Incomplete run grid')
    experiments = read_json(benchmark / 'experiments.json')
    expected = {(e['experiment'], mode) for e in experiments for mode in ['baseline', 'augmented']}
    require(set(zip(results.experiment, results['mode'])) == expected, 'Unexpected runs')
    details, histories, calibrations, raw = [], [], [], []
    for holdout, group in results.groupby('holdout', sort=True):
        predictions = []
        for _, row in group.iterrows():
            detail, pred, calibration, history, record = process_run(source, benchmark, row)
            require(detail['config_dataset_fingerprint'] == release['fingerprint'], 'Config release mismatch')
            details.append(detail)
            predictions.extend(pred)
            calibrations.extend(calibration)
            histories.append(history)
            raw.append(record)
        csv_gzip(pd.concat(predictions, ignore_index=True), output / 'predictions' / f'{holdout}.csv.gz')
        print(f'Checked {holdout}: {len(group)} runs / {len(group) * 2} predictions', flush=True)
    details = pd.DataFrame(details).sort_values(KEYS)
    results.to_csv(output / 'summary/all_results.csv', index=False)
    details.to_csv(output / 'summary/run_details.csv', index=False)
    derived_tables(details, output / 'summary')
    csv_gzip(pd.concat(histories, ignore_index=True), output / 'metadata/training_history.csv.gz')
    csv_gzip(pd.concat(calibrations, ignore_index=True), output / 'metadata/calibration_bins.csv.gz')
    with gzip.GzipFile(output / 'metadata/run_records.jsonl.gz', 'wb', mtime=0) as stream:
        stream.write(('\n'.join(json.dumps(r, sort_keys=True) for r in raw) + '\n').encode())
    statuses = [dict(read_json(p), source_status_path=p.relative_to(source).as_posix())
                for p in source.glob('*/batch_jobs/*/status.json')]
    write_json(output / 'metadata/batch_statuses.json', statuses)
    shutil.copyfile(source / 'summary/ACCEPTANCE.json', output / 'metadata/CARC_ACCEPTANCE.json')
    write_json(output / 'metadata/EXPORT_AUDIT.json', {
        'status': 'passed', 'runs': len(details), 'prediction_exports': len(details) * 2,
        'prediction_rows': len(details) * 2 * 560, 'dataset_fingerprint': release['fingerprint'],
        'manifest_hashes_labels_identities_metrics_checked': True,
        'image_bytes_and_checkpoints_rechecked_locally': False,
        'note': 'Original CARC audit checked installed data and checkpoint hashes; this export omits those binaries.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.benchmark, args.output)
