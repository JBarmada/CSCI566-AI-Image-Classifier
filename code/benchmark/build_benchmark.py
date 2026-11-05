"""Create immutable, compression-matched experiments from a verified image inventory."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

from image_inventory import Archives, GENERATORS, read_csv, scan, write_csv, write_json
from near_duplicates import candidates

COUNTS = {'train': 1400, 'val': 280, 'test': 280}
FIELDS = ['path', 'label', 'source_generator', 'source_split', 'sha256',
          'source_path', 'source_sha256', 'source_id', 'pixel_sha256']


def rng(key):
    seed = int.from_bytes(hashlib.sha256(('566:' + key).encode()).digest()[:8], 'big')
    return random.Random(seed)


def unique(rows):
    found, result = set(), []
    for row in sorted(rows, key=lambda r: r['path']):
        keys = [row[k] for k in ['source_id', 'sha256', 'pixel_sha256']]
        if found.intersection(keys):
            continue
        found.update(keys)
        result.append(row)
    return result


def choose(rows, count, key):
    if len(rows) < count:
        raise ValueError(f'{key}: need {count} distinct images, only {len(rows)} eligible')
    return rng(key).sample(sorted(rows, key=lambda r: r['path']), count)


def training_mixes(heldout, k):
    eligible = [g for g in GENERATORS if g != heldout]
    if k == 7:
        return [eligible] * 3
    choices = list(itertools.combinations(eligible, k))
    return rng(f'{heldout}:k{k}:mixes').sample(choices, 3)


def reserve_tests(store, inventory):
    lookup = {r['source_path']: r for r in inventory if int(r['label']) == 1}
    tests = {}
    for gen in GENERATORS:
        plan = store.csv(f'_release_v2/split_plans/test_{gen}.csv')
        tests[gen] = [lookup[r['path']] for r in plan if r['label'] == '1']
        if len(unique(tests[gen])) != COUNTS['test']:
            raise ValueError(f'{gen}: original fake test plan is not unique and complete')
    return tests


def partition(inventory, tests, excluded):
    real = unique([r for r in inventory if int(r['label']) == 0
                   and int(r['jpeg_match']) and r['path'] not in excluded])
    chosen = choose(real, sum(COUNTS.values()), 'shared-real')
    pools = {'real': {}}
    offset = 0
    for split, count in COUNTS.items():
        pools['real'][split] = chosen[offset:offset + count]
        offset += count
    reserved = {r[k] for rs in tests.values() for r in rs
                for k in ['source_id', 'sha256', 'pixel_sha256']}
    for gen in GENERATORS:
        available = unique([r for r in inventory if int(r['label']) == 1
                            and r['source_generator'] == gen and r['path'] not in excluded
                            and not reserved.intersection(r[k] for k in
                                                         ['source_id', 'sha256', 'pixel_sha256'])])
        chosen = choose(available, 1680, 'fake-bank:' + gen)
        pools[gen] = {'train': chosen[:1400], 'val': chosen[1400:], 'test': tests[gen]}
    return pools


def export_experiments(pools, output):
    experiments = []
    for heldout in GENERATORS:
        for k in [1, 2, 4, 7]:
            mixes = training_mixes(heldout, k)
            for rep in range(3):
                name = f'holdout_{heldout}_k{k}_rep{rep}'
                gens = sorted(mixes[rep])
                for split, count in COUNTS.items():
                    fake = pools[heldout]['test'] if split == 'test' else [
                        r for gen in gens for r in choose(pools[gen][split], count // k,
                                                         f'{name}:{gen}:{split}')]
                    rows = [{key: r[key] for key in FIELDS}
                            for r in pools['real'][split] + fake]
                    rng(name + ':' + split + ':order').shuffle(rows)
                    write_csv(output / 'manifests' / name / f'{split}.csv', rows)
                experiments.append({'experiment': name, 'holdout': heldout, 'k': k,
                                    'rep': rep, 'train_generators': gens})
    return experiments


def build(store, inventory, output):
    if (output / 'RELEASE.json').exists():
        raise FileExistsError('Benchmark already frozen; verify it instead of rebuilding in place.')
    output.mkdir(parents=True, exist_ok=True)
    eligible = unique([r for r in inventory if int(r['label']) or int(r['jpeg_match'])])
    print('Finding perceptual similarity candidates...', flush=True)
    pairs = candidates(eligible)
    write_csv(output / 'audits/near_duplicate_candidates.csv', pairs)
    tests = reserve_tests(store, inventory)
    test_paths = {r['path'] for rs in tests.values() for r in rs}
    # Conservative exclusion avoids claiming perceptual hashes prove duplication.
    # Reserved fake tests remain fixed; all other flagged images are excluded.
    excluded = {p[key] for p in pairs for key in ['left', 'right']} - test_paths
    pools = partition(inventory, tests, excluded)
    used_tests = {r['path'] for rs in tests.values() for r in rs}
    within_test = [p for p in pairs if p['left'] in used_tests and p['right'] in used_tests]
    write_csv(output / 'audits/reserved_test_candidates.csv', within_test)
    write_csv(output / 'audits/excluded_similarity.csv', [{'path': p} for p in sorted(excluded)])
    records = [dict(r, role=split, pool=pool) for pool, splits in pools.items()
               for split, rows in splits.items() for r in rows]
    write_csv(output / 'image_inventory.csv', records)
    experiments = export_experiments(pools, output)
    write_json(output / 'experiments.json', experiments)
    hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob('*')) if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    release = {'name': 'benchmark', 'fingerprint': fingerprint, 'files': hashes,
               'counts_per_class': COUNTS, 'generators': GENERATORS,
               'experiments': len(experiments), 'seed': 566,
               'jpeg': {'quality_tables': 96, 'subsampling': 0},
               'excluded_similarity_images': len(excluded),
               'reserved_test_similarity_pairs': len(within_test),
               'limitations': ['JPEG history, image size and semantic content are not matched.',
                              'Perceptual-hash filtering is conservative, not exhaustive.']}
    write_json(output / 'RELEASE.json', release)
    print(json.dumps({k: v for k, v in release.items() if k != 'files'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--release-zip', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    store = Archives(args.archives, args.release_zip)
    rows = read_csv(args.inventory.read_bytes()) if args.inventory.exists() else scan(store, args.inventory)
    build(store, rows, args.output)


if __name__ == '__main__':
    main()
