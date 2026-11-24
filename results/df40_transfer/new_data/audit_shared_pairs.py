"""Inventory FF pairs shared by four methods without inspecting model scores."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXTRA = ROOT.parents[2] / 'tmp' / 'new_method_metadata'
METHODS = ('simswap', 'fomm', 'inswap', 'pirender')


def load(method):
    directory = EXTRA if method in ('inswap', 'pirender') else ROOT / 'ff_metadata'
    path = directory / f'{method}_ff.json'
    return json.loads(path.read_text(encoding='utf-8'))[f'{method}_ff'], hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    used = set()
    for name in ('ff_split_plan.json', 'adaptation_plan.json'):
        for sample in json.loads((ROOT / name).read_text())['samples']:
            used.update(sample.get('source_ids', []))
    data = {method: load(method) for method in (*METHODS, 'wav2lip')}
    sets = []
    for method in METHODS:
        groups = data[method][0][f'{method}_Fake']['train']
        pairs = {key.removesuffix('_' + method) for key in groups}
        sets.append({p for p in pairs if re.fullmatch(r'\d{3}_\d{3}', p)})
    shared = set.intersection(*sets)
    real = data['simswap'][0]['simswap_Real']['train']
    eligible = sorted(p for p in shared if set(p.split('_')).isdisjoint(used)
                      and all(i + '_simswap' in real for i in p.split('_')))
    report = {'used_recorded_source_ids': len(used), 'common_four_method_pairs': len(shared),
              'candidate_pairs_with_unused_parsed_ids_and_real_metadata': len(eligible),
              'metadata_sha256': {k: v[1] for k, v in data.items()},
              'limitations': ['Candidate names do not prove identity or source-target roles.',
                              'Archive membership and previous image hash overlap remain to check.',
                              'No model scores were used.']}
    (ROOT / 'shared_pair_inventory.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
