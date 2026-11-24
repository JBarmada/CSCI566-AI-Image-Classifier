"""Preselect an unscored, video-ID-separated five-method candidate manifest."""
import hashlib
import json
from pathlib import Path
import re

from audit_shared_pairs import METHODS, ROOT, load


def rank(value):
    return hashlib.sha256(('566:fresh-pairs-v1:' + value).encode()).hexdigest()


def frame(item):
    paths = sorted(item['frames'], key=lambda p: (int(Path(p).stem), p))
    return paths[(len(paths) - 1) // 2]


def eligible_pairs(data, used):
    pairs = []
    for method in METHODS:
        groups = data[method][f'{method}_Fake']['train']
        pairs.append({g.removesuffix('_' + method) for g in groups})
    real = data['simswap']['simswap_Real']['train']
    return [pair for pair in set.intersection(*pairs)
            if re.fullmatch(r'\d{3}_\d{3}', pair)
            and len(set(pair.split('_'))) == 2
            and set(pair.split('_')).isdisjoint(used)
            and all(video + '_simswap' in real for video in pair.split('_'))]


def select_pairs(data, used):
    wav = data['wav2lip']['wav2lip_Fake']['train']
    selected, selected_ids = [], set()
    for pair in sorted(eligible_pairs(data, used), key=rank):
        ids = set(pair.split('_'))
        choices = [g for g in wav if g[:3] in ids and re.match(r'^\d{3}_', g)]
        if not ids.isdisjoint(selected_ids) or not choices:
            continue
        selected.append((pair, sorted(choices, key=rank)[0]))
        selected_ids.update(ids)
        if len(selected) == 30:
            break
    if len(selected) != 30:
        raise ValueError(f'Only {len(selected)} disjoint candidate pairs found')
    return selected


def build_rows(data, pairs):
    rows = []
    for pair, wavgroup in pairs:
        ids = pair.split('_')
        for video in ids:
            item = data['simswap']['simswap_Real']['train'][video + '_simswap']
            rows.append({'pair': pair, 'method': 'real', 'group': video,
                         'source_ids': [video], 'metadata_path': frame(item)})
        for method in METHODS:
            group = pair + '_' + method
            item = data[method][f'{method}_Fake']['train'][group]
            rows.append({'pair': pair, 'method': method, 'group': group,
                         'source_ids': ids, 'metadata_path': frame(item)})
        item = data['wav2lip']['wav2lip_Fake']['train'][wavgroup]
        rows.append({'pair': pair, 'method': 'wav2lip', 'group': wavgroup,
                     'source_ids': [wavgroup[:3]], 'metadata_path': frame(item)})
    return rows


def main():
    used = set()
    for name in ('ff_split_plan.json', 'adaptation_plan.json'):
        for sample in json.loads((ROOT / name).read_text())['samples']:
            used.update(sample.get('source_ids', []))
    loaded = {method: load(method) for method in (*METHODS, 'wav2lip')}
    data = {method: pair[0] for method, pair in loaded.items()}
    rows = build_rows(data, select_pairs(data, used))
    result = {'status': 'candidate only; do not score before provenance and protocol review',
              'selection': 'SHA-256 ranked 30 video-ID-disjoint shared pairs; lower-middle numeric frame',
              'metadata_sha256': {method: pair[1] for method, pair in loaded.items()},
              'limitations': ['Person and cross-dataset independence unknown',
                              'Wav2Lip audio source unknown', 'Archive membership and image hashes pending'],
              'samples': rows}
    out = ROOT / 'fresh_candidate_plan_v1.json'
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(f'Wrote {len(rows)} candidate rows; no model scores read')


if __name__ == '__main__':
    main()
