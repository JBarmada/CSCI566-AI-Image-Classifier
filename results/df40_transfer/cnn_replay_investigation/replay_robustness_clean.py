"""Run the unchanged robustness notebook clean condition on originally V100 groups."""
import json
from pathlib import Path
from IPython.display import display


def main():
    notebook = json.loads(Path('CNN_K7_Robustness_Evaluation.ipynb').read_text())
    namespace = globals()
    for index in (2, 4, 6, 8, 10, 12):
        source = ''.join(notebook['cells'][index]['source'])
        exec(compile(source, f'<notebook-cell-{index}>', 'exec'), namespace)
        if index == 4:
            namespace['HOLDOUTS'] = ['stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
                                     'VQDM', 'wukong']
            namespace['CONDITIONS'] = ['clean']
            namespace['CFG'].output_root = str(Path.cwd() / 'robustness_clean')
    print('COMPLETED robust clean replay', flush=True)


if __name__ == '__main__':
    main()
