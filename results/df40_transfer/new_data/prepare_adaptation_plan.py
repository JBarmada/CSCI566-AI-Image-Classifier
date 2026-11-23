"""Predeclare a small FF training and development sample for a CLIP head."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
METHODS = ('simswap', 'fomm', 'wav2lip')


def rank(text):
    return hashlib.sha256(('566:adaptation:' + text).encode()).digest()


def identities(name):
    return set(re.findall(r'(?:^|_)(\d{3})(?=_|$)', name))


def main():
    data = {m: json.loads((ROOT / 'ff_metadata' / (m + '_ff.json')).read_text())[m + '_ff']
            for m in METHODS}
    reference = data['simswap']['simswap_Real']
    heldout = set(reference['test'])
    real = reference['train']
    ids = sorted({g.removesuffix('_simswap') for g in real} - heldout, key=rank)
    if len(ids) < 400:
        raise ValueError('Insufficient eligible training videos')
    split = {v: ('train' if i % 4 else 'validation') for i, v in enumerate(ids)}
    chosen = []
    for group, item in real.items():
        video = group.removesuffix('_simswap')
        if video in split:
            chosen.append({'group': group, 'source_ids': [video], 'split': split[video],
                           'method': 'real', 'frames': item['frames']})
    candidates = {'real': chosen}
    for method in METHODS:
        candidates[method] = []
        for group, item in data[method][method + '_Fake']['train'].items():
            source_ids = identities(group.removesuffix('_' + method))
            if not source_ids or not source_ids <= split.keys():
                continue
            if len({split[x] for x in source_ids}) != 1:
                continue
            candidates[method].append({'group': group, 'source_ids': sorted(source_ids),
                                       'split': split[next(iter(source_ids))],
                                       'method': method, 'frames': item['frames']})
    output = []
    for split_name in ('train', 'validation'):
        for method in ('real', *METHODS):
            count = (120 if method == 'real' else 40) if split_name == 'train' else (60 if method == 'real' else 20)
            options = sorted((r for r in candidates[method] if r['split'] == split_name),
                             key=lambda r: rank(method + ':' + r['group']))
            if len(options) < count:
                raise ValueError(f'Insufficient candidates for {split_name}/{method}: {len(options)}')
            for row in options[:count]:
                frames = sorted(row.pop('frames'), key=lambda path: int(Path(path).stem))
                row['metadata_path'] = frames[(len(frames)-1)//2]
                output.append(row)
    result = {'seed': 566, 'samples': output,
              'notes': 'Source IDs partitioned before sampling; all IDs from original FF test excluded. '
                       'ID parsing and identity separation are not independently verified.'}
    with (ROOT / 'adaptation_plan.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print('Frozen plan:', len(output), 'images')


if __name__ == '__main__':
    main()
