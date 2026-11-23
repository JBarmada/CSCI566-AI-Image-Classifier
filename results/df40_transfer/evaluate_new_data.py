"""Score one independently prepared DF40 manifest with the saved models."""
import json
import sys
from pathlib import Path

import torch
import evaluate_shared as ev


def load_rows(root):
    rows = ev.vit.read_csv(root / 'manifest.csv')
    if not rows or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Missing samples or duplicate image hashes')
    for row in rows:
        row['image_id'] = row['sha256']
        row['path'] = str(root / row['path'])
    if {r['method'] for r in rows} != {'real', 'simswap', 'fomm', 'wav2lip'}:
        raise ValueError('All four sample groups required')
    return rows


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = load_rows(root)
    smoke = ev.smoke_indices(rows)
    if len(smoke) != 24:
        raise ValueError('At least six images in every group are required')
    report = {'images': len(rows), 'training': False, 'gpu': torch.cuda.get_device_name(0),
              'manifest_sha256': ev.vit.digest(root / 'manifest.csv'), 'models': {}}
    output = []
    with torch.inference_mode():
        for mode in ('baseline', 'augmented'):
            scores, checks = ev.cnn_run(mode, rows, smoke)
            report['models']['cnn_' + mode] = checks
            output.extend({**r, 'model': 'cnn', 'mode': mode, 'prob_fake': p}
                          for r, p in zip(rows, scores))
        scores, checks = ev.vit_run(rows, smoke)
        report['models']['vit'] = checks
        output.extend({**r, 'model': 'vit'} for r in scores)
    if len(output) != len(rows) * 4:
        raise ValueError('Incomplete prediction count')
    ev.vit.write_csv(out / 'predictions.csv', output)
    report['status'] = 'passed'
    (out / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
