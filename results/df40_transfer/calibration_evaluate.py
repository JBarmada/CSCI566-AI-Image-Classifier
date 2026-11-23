"""Fit validation-only thresholds and evaluate an independently supplied test set."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

METHODS = ('simswap', 'fomm', 'wav2lip')
FIELDS = ('sha256', 'video_group', 'method', 'label')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(group):
    return set(re.findall(r'(?<![A-Za-z0-9])id\d+(?!\d)', group.lower()))


def unique_images(rows):
    images = {}
    for row in rows:
        for field in FIELDS:
            if not row.get(field):
                raise ValueError(f'Missing {field}')
        sha = row['sha256'].lower()
        if not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError('Invalid SHA-256')
        if row['method'] not in ('real', *METHODS):
            raise ValueError('Unexpected method')
        if row['label'] != ('0' if row['method'] == 'real' else '1'):
            raise ValueError('Label/method mismatch')
        identity = tuple(row[field] for field in FIELDS[1:])
        if sha in images and images[sha] != identity:
            raise ValueError('Conflicting metadata for an image hash')
        images[sha] = identity
    if not images:
        raise ValueError('Empty input')
    return images


def split_members(images):
    groups = {v[0].lower() for v in images.values()}
    known_ids = set().union(*(identities(group) for group in groups))
    unknown_groups = sorted(group for group in groups if not identities(group))
    return {'hashes': set(images), 'groups': groups, 'identities': known_ids,
            'unknown_identity_groups': unknown_groups}


def check_separation(named):
    names = list(named)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for field in ('hashes', 'groups', 'identities'):
                overlap = named[left][field] & named[right][field]
                if overlap:
                    raise ValueError(f'{left}/{right} share {field}: {sorted(overlap)[:5]}')


def scores_by_model(rows, images):
    result = {}
    for row in rows:
        key = (row.get('model'), row.get('mode'))
        if not all(key):
            raise ValueError('Model and mode required')
        probability = float(row['prob_fake'])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        bucket = result.setdefault(key, {})
        sha = row['sha256'].lower()
        if sha in bucket:
            raise ValueError('Duplicate model/image prediction')
        bucket[sha] = probability
    if not result or any(set(scores) != set(images) for scores in result.values()):
        raise ValueError('Every model must score the same complete image set')
    if set(value[1] for value in images.values()) != {'real', *METHODS}:
        raise ValueError('Real and all three methods are required')
    return result


def select_threshold(real_scores, target):
    if not real_scores or not 0 <= target < 1:
        raise ValueError('Invalid real scores or FPR target')
    # Strictly exceed an observed score because prediction uses >= threshold.
    candidates = [0.0] + [math.nextafter(score, math.inf) for score in set(real_scores)]
    eligible = [t for t in candidates if sum(p >= t for p in real_scores) / len(real_scores) <= target]
    return min(eligible)


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    denom = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denom
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def metrics(images, scores, threshold):
    real = [scores[s] for s, v in images.items() if v[1] == 'real']
    results = {}
    for method in METHODS:
        fake = [scores[s] for s, v in images.items() if v[1] == method]
        fp = sum(p >= threshold for p in real)
        tp = sum(p >= threshold for p in fake)
        results[method] = {'real': len(real), 'fake': len(fake), 'fp': fp,
                           'tn': len(real) - fp, 'tp': tp, 'fn': len(fake) - tp,
                           'real_fpr': fp / len(real), 'real_fpr_wilson95': wilson(fp, len(real)),
                           'fake_recall': tp / len(fake), 'fake_recall_wilson95': wilson(tp, len(fake)),
                           'balanced_accuracy': ((1 - fp / len(real)) + tp / len(fake)) / 2}
    return results


def serial_members(members):
    return {key: sorted(value) for key, value in members.items()}


def fit_thresholds(validation, old, target):
    images = unique_images(validation)
    members = {'validation': split_members(images), 'previous_test': split_members(unique_images(old))}
    check_separation(members)
    scores = scores_by_model(validation, images)
    output = []
    for key, score in sorted(scores.items()):
        real = [score[s] for s, v in images.items() if v[1] == 'real']
        threshold = select_threshold(real, target)
        output.append({'model': key[0], 'mode': key[1], 'threshold': threshold,
                       'validation': metrics(images, score, threshold)})
    return {'target_validation_fpr': target, 'models': output,
            'validation_members': serial_members(members['validation']),
            'unresolved_validation_identity_groups': members['validation']['unknown_identity_groups'],
            'policy': 'Lowest threshold achieving validation-real FPR <= 0.10; fixed separately for each model.'}


def evaluate_frozen(frozen, test, old):
    images = unique_images(test)
    val = frozen['validation_members']
    members = {'validation': {k: set(v) for k, v in val.items()},
               'fresh_test': split_members(images), 'previous_test': split_members(unique_images(old))}
    check_separation(members)
    scores = scores_by_model(test, images)
    thresholds = {(r['model'], r['mode']): r['threshold'] for r in frozen['models']}
    if set(thresholds) != set(scores):
        raise ValueError('Frozen model configurations differ from test predictions')
    output = []
    for key, score in sorted(scores.items()):
        threshold = float(thresholds[key])
        if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
            raise ValueError('Invalid frozen threshold')
        output.append({'model': key[0], 'mode': key[1], 'threshold': threshold,
                       'fresh_test': metrics(images, score, threshold),
                       'fresh_test_threshold_0_5': metrics(images, score, .5)})
    return {'models': output,
            'unresolved_identity_groups': {k: sorted(v['unknown_identity_groups']) for k, v in members.items()},
            'separation_limit': 'Exact hash, group and recognizable id-token checks only; unknown or renamed identities and linked sources are not proved disjoint.',
            'uncertainty_limit': 'Wilson intervals are descriptive image-level binomial intervals; identity and source dependence weaken independence assumptions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--validation', type=Path, required=True)
    test = commands.add_parser('test')
    test.add_argument('--frozen', type=Path, required=True)
    test.add_argument('--test', type=Path, required=True)
    for command in (fit, test):
        command.add_argument('--previous-test', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        report = fit_thresholds(read_csv(args.validation), read_csv(args.previous_test), 0.10)
        report['input_sha256'] = {key: file_hash(getattr(args, key)) for key in ('validation', 'previous_test')}
    else:
        frozen = json.loads(args.frozen.read_text(encoding='utf-8'))
        if frozen['input_sha256']['previous_test'] != file_hash(args.previous_test):
            raise ValueError('Previous-test manifest differs from frozen artifact')
        report = evaluate_frozen(frozen, read_csv(args.test), read_csv(args.previous_test))
        report['input_sha256'] = {key: file_hash(getattr(args, key)) for key in ('frozen', 'test', 'previous_test')}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
