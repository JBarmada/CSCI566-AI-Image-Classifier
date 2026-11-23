"""Count candidate CDF groups after conservative exclusions."""
import csv
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'shared_dataset_v1/manifest.csv'
BASE = 'https://huggingface.co/datasets/sinister007/df40_dataset/resolve/a203e9b880d69339178f1a6aacd937b07a8ef199/dataset_json/'


def ids(group):
    return set(re.findall(r'id\d+', group))


def main():
    rows = list(csv.DictReader(DATA.open(newline='', encoding='utf-8')))
    used_groups = {r['video_group'] for r in rows}
    used_ids = set().union(*(ids(g) for g in used_groups))
    result = {}
    for method in ('simswap', 'fomm', 'wav2lip'):
        with urllib.request.urlopen(BASE + method + '_cdf.json', timeout=60) as response:
            data = json.load(response)[method + '_cdf']
        for label in ('Real', 'Fake'):
            groups = data[method + '_' + label]['test']
            unseen_group = [g for g in groups if g not in used_groups]
            known_disjoint = [g for g in unseen_group if ids(g) and not (ids(g) & used_ids)]
            unknown = [g for g in unseen_group if not ids(g)]
            result[method + '_' + label] = {'total_groups': len(groups),
                'group_names_not_in_existing_test': len(unseen_group),
                'known_id_tokens_not_in_existing_test': len(known_disjoint),
                'unknown_id_group_names': len(unknown),
                'known_id_disjoint_groups': known_disjoint}
    result['existing_test_known_id_count'] = len(used_ids)
    result['limitations'] = ['Group name IDs are not a verified source-target mapping.',
                             'Numeric YouTube group identities are unknown.',
                             'Official CDF val and test metadata contain identical paths.']
    (ROOT / 'CANDIDATE_COUNTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
