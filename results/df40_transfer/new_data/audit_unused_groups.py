"""Count potential FF groups without claiming person-level independence."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
METHODS = ('simswap', 'fomm', 'wav2lip')


def used_ids():
    ids = set()
    for name in ('ff_split_plan.json', 'adaptation_plan.json'):
        plan = json.loads((ROOT / name).read_text(encoding='utf-8'))
        for row in plan['samples']:
            ids.update(row.get('source_ids', []))
    return ids


def main():
    used = used_ids()
    output = {'known_used_source_ids': len(used), 'methods': {},
              'limitation': 'Filename IDs are incomplete proxies; identity and source roles are unverified.'}
    for method in METHODS:
        meta = json.loads((ROOT / 'ff_metadata' / f'{method}_ff.json').read_text())[f'{method}_ff']
        output['methods'][method] = {}
        for label in ('Real', 'Fake'):
            groups = meta[f'{method}_{label}']['train']
            parsed = [set(re.findall(r'(?:^|_)(\d{3})(?=_|$)', name)) for name in groups]
            possible = sum(bool(ids) and ids.isdisjoint(used) for ids in parsed)
            output['methods'][method][label.lower()] = {'groups': len(groups),
                'no_known_used_parsed_ids': possible, 'no_parsed_ids': sum(not x for x in parsed)}
    (ROOT / 'unused_group_inventory.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
