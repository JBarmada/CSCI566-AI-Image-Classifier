"""Repeat clean CNN inference to diagnose an audit discrepancy."""
import json
from pathlib import Path

from IPython.display import display


def main():
    notebook = json.loads(Path('CNN_K7_Robustness_Evaluation.ipynb').read_text())
    namespace = globals()
    for index in (2, 4, 6, 8, 10, 12):
        source = ''.join(notebook['cells'][index]['source'])
        exec(compile(source, f'<notebook-cell-{index}>', 'exec'), namespace)
        if index == 2:
            namespace['torch'].backends.cudnn.deterministic = True
            namespace['torch'].backends.cudnn.benchmark = False
        if index == 4:
            namespace['HOLDOUTS'] = ['stable_diffusion_v_1_4']
            namespace['CONDITIONS'] = ['clean']
            namespace['CFG'].output_root = str(Path.cwd() / 'diagnostic_clean_sd14_deterministic')


if __name__ == '__main__':
    main()
