"""Run the existing CNN robustness notebook for one isolated holdout."""
import json
import sys
from pathlib import Path
from IPython.display import display

HOLDOUTS = ('BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
            'stable_diffusion_v_1_5', 'VQDM', 'wukong')


def main():
    holdout = HOLDOUTS[int(sys.argv[1])]
    notebook = json.loads(Path('CNN_K7_Robustness_Evaluation.ipynb').read_text())
    namespace = globals()
    for index in (2, 4, 6, 8, 10, 12, 14):
        source = ''.join(notebook['cells'][index]['source'])
        exec(compile(source, f'<notebook-cell-{index}>', 'exec'), namespace)
        if index == 4:
            namespace['HOLDOUTS'] = [holdout]
            namespace['CFG'].output_root = str(Path.cwd() / 'outputs' / holdout)
    print('Completed holdout:', holdout, flush=True)


if __name__ == '__main__':
    main()
