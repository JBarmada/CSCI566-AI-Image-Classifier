"""Independently validate a frozen benchmark and optionally its installed image bytes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from image_inventory import GENERATORS, digest, properties, read_csv, write_json

COUNTS = {'train': 1400, 'val': 280, 'test': 280}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_release(root):
    release = json.loads((root / 'RELEASE.json').read_text())
    actual = {}
    for relative, sha in release['files'].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), 'Release path escapes benchmark')
        actual[relative] = digest(path.read_bytes())
        require(actual[relative] == sha, f'Changed release file: {relative}')
    fingerprint = digest(json.dumps(actual, sort_keys=True).encode())
    require(fingerprint == release['fingerprint'], 'Invalid release fingerprint')
    return release


def check_experiment(root, experiment, inventory, real_sets, test_sets, roles):
    name, heldout, k = experiment['experiment'], experiment['holdout'], experiment['k']
    seen = {key: set() for key in ['sha256', 'source_id', 'pixel_sha256']}
    for split, count in COUNTS.items():
        rows = read_csv((root / 'manifests' / name / f'{split}.csv').read_bytes())
        require(Counter(int(r['label']) for r in rows) == {0: count, 1: count}, f'{name}/{split}: counts')
        for key in seen:
            values = [r[key] for r in rows]
            require(len(set(values)) == len(values), f'{name}/{split}: duplicate {key}')
            require(not seen[key].intersection(values), f'{name}/{split}: split leakage {key}')
            seen[key].update(values)
        real = frozenset(r['source_id'] for r in rows if r['label'] == '0')
        require(real == real_sets.setdefault(split, real), f'{name}: real {split} changed')
        fake = [r for r in rows if r['label'] == '1']
        generators = {r['source_generator'] for r in fake}
        expected = {heldout} if split == 'test' else set(experiment['train_generators'])
        require(generators == expected, f'{name}/{split}: wrong generators')
        if split != 'test':
            require(heldout not in generators and len(generators) == k, f'{name}: holdout leakage')
            require(set(Counter(r['source_generator'] for r in fake).values()) == {count // k},
                    f'{name}/{split}: unequal generator allocation')
        else:
            ids = frozenset(r['sha256'] for r in rows)
            require(ids == test_sets.setdefault(heldout, ids), f'{name}: test membership changed')
        for r in rows:
            require(r['path'] in inventory, f'{name}: unknown path')
            original = inventory[r['path']]
            require(all(r[key] == original[key] for key in r), f'{name}: row provenance changed')
            require(original['role'] == split, f'{name}: image changed split role')
            roles[r['source_id']].add(split)


def verify_images(data_root, inventory):
    for i, row in enumerate(inventory.values(), 1):
        path = (data_root / row['path']).resolve()
        require(path.is_relative_to(data_root.resolve()), 'Image path escapes data root')
        data = path.read_bytes()
        require(digest(data) == row['sha256'], f'Image hash mismatch: {path}')
        props = properties(data)
        require(props['jpeg_match'], f'JPEG settings mismatch: {path}')
        require(props['pixel_sha256'] == row['pixel_sha256'], f'Decoded pixels changed: {path}')
        source = (data_root / row['source_path']).resolve()
        require(source.is_relative_to(data_root.resolve()), 'Source path escapes data root')
        require(digest(source.read_bytes()) == row['source_sha256'], f'Source mismatch: {source}')
        if i % 1000 == 0:
            print(f'Installed image audit: {i}/{len(inventory)}', flush=True)


def audit(root, data_root=None):
    release = verify_release(root)
    inventory = {r['path']: r for r in read_csv((root / 'image_inventory.csv').read_bytes())}
    require(len(inventory) == 17640, 'Unexpected unique image count')
    require(all(r['jpeg_match'] == '1' for r in inventory.values()), 'Unmatched JPEG selected')
    require(all(r['path'].startswith('_release_v2/jpeg_ai/') for r in inventory.values()
                if r['label'] == '1'), 'Original fake PNG selected')
    experiments = json.loads((root / 'experiments.json').read_text())
    expected = {f'holdout_{g}_k{k}_rep{r}' for g in GENERATORS for k in [1, 2, 4, 7] for r in range(3)}
    require(len(experiments) == 96 and {e['experiment'] for e in experiments} == expected,
            'Incomplete experiment grid')
    for holdout in GENERATORS:
        for k in [1, 2, 4]:
            mixes = {tuple(e['train_generators']) for e in experiments
                     if e['holdout'] == holdout and e['k'] == k}
            require(len(mixes) == 3, f'{holdout}/k{k}: repeated generator combinations')
    real_sets, test_sets, roles = {}, {}, defaultdict(set)
    for experiment in experiments:
        check_experiment(root, experiment, inventory, real_sets, test_sets, roles)
    require(all(len(v) == 1 for v in roles.values()), 'Image roles change across experiments')
    suspicious = read_csv((root / 'audits/near_duplicate_candidates.csv').read_bytes())
    for pair in suspicious:
        if 'left' not in pair:
            continue
        a, b = inventory.get(pair['left']), inventory.get(pair['right'])
        if a and b:
            require(a['role'] == b['role'] == 'test', 'Similarity candidate crosses split roles')
            require(a['source_generator'] != b['source_generator'], 'Unreviewed similarity within a test')
    if data_root:
        verify_images(data_root, inventory)
    return {'status': 'passed', 'fingerprint': release['fingerprint'], 'experiments': 96,
            'planned_training_runs': 192, 'images': len(inventory), 'counts_per_class': COUNTS,
            'installed_image_bytes_verified': data_root is not None,
            'fixed_test_sets': True, 'fixed_real_sets': True, 'heldout_exclusion': True,
            'global_split_roles_fixed': True, 'jpeg_settings_matched': True,
            'distinct_generator_mixes_for_k1_k2_k4': True,
            'near_duplicate_cross_split_candidates': 0, 'limitations': release['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.benchmark, args.data_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
