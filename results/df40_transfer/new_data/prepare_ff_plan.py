"""Make a deterministic FaceForensics++ video-group split plan."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ('simswap', 'fomm', 'wav2lip')


def load_metadata():
    return {m: json.loads((ROOT / 'ff_metadata' / f'{m}_ff.json').read_text())[f'{m}_ff']
            for m in METHODS}


def group_split(data):
    real = set(data['simswap']['simswap_Real']['test'])
    pairs = set()
    for group in data['simswap']['simswap_Fake']['test']:
        parts = group.split('_')
        if len(parts) != 2 or not set(parts) <= real:
            raise ValueError(f'Unexpected group: {group}')
        pairs.add(tuple(sorted(parts)))
    if len(real) != 140 or len(pairs) != 70 or len({x for pair in pairs for x in pair}) != 140:
        raise ValueError('Expected 70 disjoint original-video pairs')
    ordered = sorted(pairs, key=lambda p: hashlib.sha256(('566:' + ':'.join(p)).encode()).digest())
    return {node: ('validation' if i < 35 else 'test')
            for i, pair in enumerate(ordered) for node in pair}


def main():
    data = load_metadata()
    split = group_split(data)
    plan = []
    for method in ('real', *METHODS):
        source = 'simswap' if method == 'real' else method
        label = 'Real' if method == 'real' else 'Fake'
        group = data[source][source + '_' + label]
        if group['test'] != group['val']:
            raise ValueError('Previously observed val/test alias changed')
        for name, item in sorted(group['test'].items()):
            parts = name.split('_')
            role = split[parts[0]]
            if method in ('simswap', 'fomm') and split.get(parts[1]) != role:
                raise ValueError(f'Fake pair crosses split: {name}')
            frames = sorted(item['frames'], key=lambda x: int(Path(x).stem))
            plan.append({'split': role, 'method': method, 'group': name,
                         'metadata_path': frames[(len(frames)-1)//2],
                         'source_ids': parts[:2] if method in ('simswap', 'fomm') else parts[:1]})
    counts = Counter((x['split'], x['method']) for x in plan)
    if any(counts[(s, m)] != 70 for s in ('validation', 'test') for m in ('real', *METHODS)):
        raise ValueError(f'Unexpected counts: {counts}')
    dest = ROOT / 'ff_split_plan.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps({'seed': 566, 'selection': 'hash-ranked 70 linked video pairs, split 35/35; lower-middle numeric frame',
                                'limitations': 'Video ID split, identities and renamed clips unverified', 'samples': plan}, indent=2))
    print('Created plan:', dict(counts))


if __name__ == '__main__':
    main()
