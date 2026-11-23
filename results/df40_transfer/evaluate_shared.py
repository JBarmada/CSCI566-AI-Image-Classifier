"""Inference-only replay, smoke, and fixed shared-set evaluation."""
import csv
import json
import sys
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode

import smoke_inference as vit

ROOT = vit.SHARED
DATA = ROOT / 'data/df40_shared_v1'
OUT = Path(sys.argv[1])


def rows_from_manifest():
    rows = vit.read_csv(DATA / 'manifest.csv')
    for row in rows:
        row['image_id'] = row['sha256']
        row['path'] = str(DATA / row['path'])
    if len(rows) != 500 or len({r['sha256'] for r in rows}) != 500:
        raise ValueError('Unexpected shared-set manifest')
    return rows


def smoke_indices(rows):
    return [i for method in ('real', 'simswap', 'fomm', 'wav2lip')
            for i in [j for j, row in enumerate(rows) if row['method'] == method][:6]]


def cnn_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def cnn_scores(model, rows, transform):
    scores = []
    for start in range(0, len(rows), 64):
        tensors = []
        for row in rows[start:start + 64]:
            path = Path(row['path'])
            if vit.digest(path) != row['image_id']:
                raise ValueError('Image hash mismatch: ' + str(path))
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
        with torch.autocast(device_type='cuda', enabled=True):
            scores.extend(torch.sigmoid(model(torch.stack(tensors).cuda()).squeeze(1)).cpu().tolist())
    if not all(0 <= value <= 1 for value in scores):
        raise ValueError('Invalid CNN probability')
    return scores


def cnn_run(mode, rows, smoke):
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
    model.cuda().eval().requires_grad_(False)
    reference = vit.read_csv(folder / 'test_predictions.csv')[:64]
    ref_rows = [{'path': str(ROOT / 'data/extracted/GenImage_Compact' / r['path']),
                 'image_id': r['sha256']} for r in reference]
    transform = cnn_transform()
    ref_scores = cnn_scores(model, ref_rows, transform)
    diff = max(abs(a - float(b['probability_fake'])) for a, b in zip(ref_scores, reference))
    if diff > 0.001:
        raise ValueError(f'CNN original reference replay failed: {mode} {diff}')
    smoke_rows = [rows[i] for i in smoke]
    first = cnn_scores(model, smoke_rows, transform)
    second = cnn_scores(model, smoke_rows, transform)
    repeat = max(abs(a - b) for a, b in zip(first, second))
    if repeat > 0.001:
        raise ValueError('CNN repeat check failed')
    scores = cnn_scores(model, rows, transform)
    if vit.digest(checkpoint) != marker['checkpoint_sha256']:
        raise ValueError('CNN checkpoint changed')
    return scores, {'reference_max_difference': diff, 'repeat_max_difference': repeat,
                    'checkpoint_sha256': marker['checkpoint_sha256']}


def vit_run(rows, smoke):
    runs = vit.load_runs()
    weights = Path.home() / '.cache/clip/ViT-L-14.pt'
    original_hash = vit.digest(weights)
    encoder, transform = clip.load(str(weights), device='cuda', jit=False)
    encoder.eval().requires_grad_(False)
    references = vit.reference_rows(runs)
    _, deltas = vit.predict(references, vit.encode(references, encoder, transform), runs, True)
    if any(value > 1e-5 for value in deltas.values()):
        raise ValueError('ViT original reference replay failed')
    smoke_rows = [rows[i] for i in smoke]
    first, _ = vit.predict(smoke_rows, vit.encode(smoke_rows, encoder, transform), runs)
    second, _ = vit.predict(smoke_rows, vit.encode(smoke_rows, encoder, transform), runs)
    repeat = max(abs(a['prob_fake'] - b['prob_fake']) for a, b in zip(first, second))
    if repeat > 1e-6:
        raise ValueError('ViT repeat check failed')
    output, _ = vit.predict(rows, vit.encode(rows, encoder, transform), runs)
    if vit.digest(weights) != original_hash or any(vit.digest(r['checkpoint']) != r['sha256'] for r in runs):
        raise ValueError('ViT model changed')
    checks = {'reference_max_differences': deltas, 'repeat_max_difference': repeat,
              'backbone_sha256': original_hash,
              'checkpoint_sha256': {r['mode']: r['sha256'] for r in runs}}
    return output, checks


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = rows_from_manifest()
    smoke = smoke_indices(rows)
    if len(smoke) != 24:
        raise ValueError('Expected 24 smoke images')
    report = {'smoke_images': 24, 'test_images': len(rows),
              'smoke_policy': 'First six per method in frozen manifest; included in full descriptive test; no score-based tuning',
              'gpu': torch.cuda.get_device_name(0), 'training': False, 'models': {}}
    output = []
    with torch.inference_mode():
        for mode in ('baseline', 'augmented'):
            scores, check = cnn_run(mode, rows, smoke)
            report['models']['cnn_' + mode] = check
            output.extend({**r, 'model': 'cnn', 'mode': mode, 'prob_fake': p,
                           'smoke': i in smoke} for i, (r, p) in enumerate(zip(rows, scores)))
        predictions, checks = vit_run(rows, smoke)
        report['models']['vit'] = checks
        output.extend({**r, 'model': 'vit', 'smoke': r['image_id'] in
                       {rows[i]['image_id'] for i in smoke}} for r in predictions)
    if len(output) != 2000:
        raise ValueError('Unexpected prediction count')
    vit.write_csv(OUT / 'predictions.csv', output)
    report['status'] = 'passed'
    (OUT / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
