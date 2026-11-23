"""Re-run one clean CNN checkpoint through the original benchmark functions."""
import json
from pathlib import Path

from IPython.display import display

BASE = Path('/project2/yzhao010_1531/csci566-genimage')
NAME = 'holdout_stable_diffusion_v_1_4_k7_rep0__baseline__seed17'


def main():
    notebook = json.loads(Path('cnn.ipynb').read_text())
    ns = globals()
    for index in (2, 4, 7, 13, 15):
        source = ''.join(notebook['cells'][index]['source'])
        if index == 2:
            source = source.replace('import seaborn as sns\n', '')
        exec(compile(source, f'<original-cell-{index}>', 'exec'), ns)
    ns['seed_everything'](17)
    source = BASE / 'results/cnn_benchmark/notebook/runs' / NAME
    frame = ns['pd'].read_csv(source / 'input_test.csv')
    loader = ns['make_loader'](frame, ns['eval_transform'], seed=17, shuffle=False)
    model = ns['resnet50'](weights=None)
    model.fc = ns['nn'].Linear(model.fc.in_features, 1)
    saved = ns['torch'].load(source / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    result = ns['predict'](model.to(ns['DEVICE']), loader)
    original = ns['pd'].read_csv(source / 'test_predictions.csv')
    merged = result.merge(original, on='sha256', suffixes=('_new', '_old'), validate='one_to_one')
    print('Rows:', len(merged), 'max difference:', float(
        (merged['probability_fake_new'] - merged['probability_fake_old']).abs().max()), flush=True)
    result.to_csv('original_notebook_replay.csv', index=False)


if __name__ == '__main__':
    main()
