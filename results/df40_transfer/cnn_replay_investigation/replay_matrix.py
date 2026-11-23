"""Replay original clean CNN inference on four originally V100 holdouts."""
import hashlib
import json
from pathlib import Path
from IPython.display import display

BASE = Path('/project2/yzhao010_1531/csci566-genimage')
HOLDOUTS = ('stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong')


def setup():
    notebook = json.loads(Path('cnn.ipynb').read_text())
    ns = globals()
    for index in (2, 4, 7, 13, 15):
        source = ''.join(notebook['cells'][index]['source'])
        if index == 2:
            source = source.replace('import seaborn as sns\n', '')
        exec(compile(source, f'<original-cell-{index}>', 'exec'), ns)
    return ns


def replay(ns, name):
    ns['seed_everything'](17)
    source = BASE / 'results/cnn_benchmark/notebook/runs' / name
    frame = ns['pd'].read_csv(source / 'input_test.csv')
    loader = ns['make_loader'](frame, ns['eval_transform'], seed=17, shuffle=False)
    model = ns['resnet50'](weights=None)
    model.fc = ns['nn'].Linear(model.fc.in_features, 1)
    checkpoint = source / 'best.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker = json.loads((source / 'TRAINING_COMPLETE.json').read_text())
    if digest != marker['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {name}')
    saved = ns['torch'].load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    result = ns['predict'](model.to(ns['DEVICE']), loader)
    original = ns['pd'].read_csv(source / 'test_predictions.csv')
    merged = result.merge(original, on='sha256', suffixes=('_new', '_old'), validate='one_to_one')
    if len(merged) != len(original) or len(result) != len(original):
        raise ValueError(f'Prediction row mismatch: {name}')
    difference = (merged['probability_fake_new'] - merged['probability_fake_old']).abs()
    row = {'run': name, 'rows': len(merged), 'checkpoint_sha256': digest,
           'same_order': result['sha256'].equals(original['sha256']),
           'max_abs_difference': float(difference.max()),
           'mean_abs_difference': float(difference.mean()),
           'above_0_001': int((difference > 0.001).sum()),
           'classification_changes': int(((merged['probability_fake_new'] >= 0.5) !=
                                          (merged['probability_fake_old'] >= 0.5)).sum())}
    print(json.dumps(row), flush=True)
    return row


def main():
    ns = setup()
    rows = []
    for holdout in HOLDOUTS:
        for rep in range(3):
            for mode in ('baseline', 'augmented'):
                name = f'holdout_{holdout}_k7_rep{rep}__{mode}__seed17'
                rows.append(replay(ns, name))
                Path('matrix_results.json').write_text(json.dumps(rows, indent=2))
    print('COMPLETED', len(rows), flush=True)


if __name__ == '__main__':
    main()
