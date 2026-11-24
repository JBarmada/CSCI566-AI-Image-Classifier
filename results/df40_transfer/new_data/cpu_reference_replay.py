"""Replay saved benchmark predictions on CPU without modifying GPU jobs."""
import json
from pathlib import Path
import platform
import sys
import time

import clip
from PIL import Image
import torch
from torchvision.models import resnet50

ROOT = Path('/project2/yzhao010_1531/csci566-genimage')
sys.path.insert(0, str(ROOT / 'results/df40_ff_eval_v1'))
import smoke_inference as vit
import evaluate_shared as shared


def cnn_reference(mode):
    folder = ROOT / ('results/cnn_benchmark/notebook/runs/'
                     f'holdout_ADM_k7_rep0__{mode}__seed17')
    marker = json.loads((folder / 'TRAINING_COMPLETE.json').read_text())
    checkpoint = folder / 'best.pt'
    if vit.digest(checkpoint) != marker['checkpoint_sha256']:
        raise ValueError('CNN checkpoint hash mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['protocol_hash'] != marker['protocol_hash']:
        raise ValueError('CNN checkpoint protocol mismatch')
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    rows = vit.read_csv(folder / 'test_predictions.csv')[:64]
    transform = shared.cnn_transform()
    output = []
    for start in range(0, len(rows), 16):
        tensors = []
        for row in rows[start:start + 16]:
            path = ROOT / 'data/extracted/GenImage_Compact' / row['path']
            if vit.digest(path) != row['sha256']:
                raise ValueError('Reference image hash mismatch')
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
        output.extend(torch.sigmoid(model(torch.stack(tensors)).flatten()).tolist())
    return rows, output, marker['checkpoint_sha256']


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    started = time.monotonic()
    report = {'device': 'cpu', 'training': False, 'torch': torch.__version__,
              'python': platform.python_version(), 'models': {}}
    comparisons = []
    with torch.inference_mode():
        for mode in ('baseline', 'augmented'):
            rows, scores, checkpoint_hash = cnn_reference(mode)
            diffs = [abs(p - float(row['probability_fake'])) for p, row in zip(scores, rows)]
            report['models']['cnn_' + mode] = {'max_abs_difference': max(diffs),
                'tolerance': .001, 'passed': max(diffs) <= .001,
                'images': len(rows), 'checkpoint_sha256': checkpoint_hash}
            comparisons.extend({'model': 'cnn', 'mode': mode, 'sha256': row['sha256'],
                                'reference': row['probability_fake'], 'cpu': p}
                               for row, p in zip(rows, scores))
            print(json.dumps(report['models']['cnn_' + mode]), flush=True)
        # The existing loader and encoder use .cuda(); in this isolated CPU-only
        # process redirect those transfers to CPU while retaining the same logic.
        torch.nn.Module.cuda = lambda self, *args, **kwargs: self.cpu()
        torch.Tensor.cuda = lambda self, *args, **kwargs: self.cpu()
        weights = Path.home() / '.cache/clip/ViT-L-14.pt'
        backbone_hash = vit.digest(weights)
        runs = vit.load_runs()
        encoder, transform = clip.load(str(weights), device='cpu', jit=False)
        encoder.eval().requires_grad_(False)
        rows = vit.reference_rows(runs)
        scores, differences = vit.predict(rows, vit.encode(rows, encoder, transform), runs, True)
        for run in runs:
            mode = run['mode']
            report['models']['vit_' + mode] = {'max_abs_difference': differences[mode],
                'tolerance': 1e-5, 'passed': differences[mode] <= 1e-5,
                'images': len(rows), 'checkpoint_sha256': run['sha256']}
        for row in scores:
            ref = next(run['reference'][row['image_id']]['prob_fake']
                       for run in runs if run['mode'] == row['mode'])
            comparisons.append({'model': 'vit', 'mode': row['mode'], 'sha256': row['image_id'],
                                'reference': ref, 'cpu': row['prob_fake']})
    vit.write_csv(out / 'reference_predictions.csv', comparisons)
    report.update(backbone_sha256=backbone_hash, elapsed_seconds=time.monotonic() - started,
                  status='passed' if all(v['passed'] for v in report['models'].values()) else 'mismatch')
    (out / 'REFERENCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
