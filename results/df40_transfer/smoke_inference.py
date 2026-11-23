"""Inference-only frozen CLIP smoke with an original-benchmark replay gate."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image


SHARED = Path('/project2/yzhao010_1531/csci566-genimage')
MODES = ['baseline', 'augmented']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load_runs():
    runs = []
    for mode in MODES:
        folder = SHARED / ('results/vit_benchmark/ADM/runs/'
                          f'holdout_ADM_k7_rep0__{mode}__seed17')
        config = json.loads((folder / 'config.json').read_text())
        marker = json.loads((folder / 'TRAINING_COMPLETE.json').read_text())
        checkpoint = folder / 'best_head.pt'
        checkpoint_hash = digest(checkpoint)
        if config['smoke_test'] or config['protocol_hash'] != marker['protocol_hash']:
            raise ValueError('Wrong source training identity')
        if checkpoint_hash != marker['checkpoint_sha256']:
            raise ValueError('Checkpoint bytes disagree with acceptance marker')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['protocol_hash'] != config['protocol_hash']:
            raise ValueError('Head protocol mismatch')
        head = torch.nn.Linear(saved['feature_dim'], 1).cuda().eval()
        head.load_state_dict(saved['head_state_dict'], strict=True)
        head.requires_grad_(False)
        predictions = list(folder.glob('evaluations/clean_*/test_predictions.csv'))
        if len(predictions) != 1:
            raise ValueError('Expected exactly one original clean reference')
        runs.append({'mode': mode, 'head': head, 'folder': folder,
                     'checkpoint': checkpoint, 'sha256': checkpoint_hash,
                     'reference': {r['sha256']: r for r in read_csv(predictions[0])}})
    return runs


def reference_rows(runs):
    entries = runs[0]['reference']
    ordered = list(entries.values())
    starts = set()
    for label in [0, 1]:
        matching = [i for i, r in enumerate(ordered) if int(r['label']) == label]
        starts.update((i // 16) * 16 for i in matching[:2])
    selected = []
    for start in sorted(starts):
        for row in ordered[start:start + 16]:
            selected.append({'image_id': row['sha256'], 'label': int(row['label']),
                             'path': str(SHARED / 'data/extracted/GenImage_Compact' / row['path'])})
    return selected


def encode(rows, encoder, preprocess):
    tensors = []
    for row in rows:
        path = Path(row['path'])
        if digest(path) != row['image_id']:
            raise ValueError(f'Image hash mismatch: {path}')
        with Image.open(path) as image:
            tensors.append(preprocess(image.convert('RGB')))
    results = []
    for start in range(0, len(tensors), 16):
        batch = torch.stack(tensors[start:start + 16]).cuda()
        results.append(F.normalize(encoder.encode_image(batch).float(), dim=-1))
    features = torch.cat(results)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite features')
    return features


def predict(rows, features, runs, reference=False):
    output, checks = [], {}
    for run in runs:
        scores = torch.sigmoid(run['head'](features).flatten()).cpu().tolist()
        if not all(0 <= p <= 1 for p in scores):
            raise ValueError('Invalid scores')
        differences = []
        for row, probability in zip(rows, scores):
            output.append({**row, 'mode': run['mode'], 'prob_fake': probability})
            if reference:
                old = run['reference'][row['image_id']]
                if int(old['label']) != int(row['label']):
                    raise ValueError('Reference label mismatch')
                differences.append(abs(probability - float(old['prob_fake'])))
        if reference:
            checks[run['mode']] = max(differences)
    return output, checks


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run in a GPU compute allocation')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    weights = Path.home() / '.cache/clip/ViT-L-14.pt'
    if not weights.is_file():
        raise FileNotFoundError('Existing backbone missing; downloads disabled')
    backbone_hash = digest(weights)
    runs = load_runs()
    encoder, preprocess = clip.load(str(weights), device='cuda', jit=False)
    encoder.eval().requires_grad_(False)
    with torch.inference_mode():
        references = reference_rows(runs)
        predictions, deltas = predict(references, encode(references, encoder, preprocess), runs, True)
        write_csv(args.output / 'reference_predictions.csv', predictions)
        report = {'reference_max_abs_differences': deltas, 'reference_tolerance': 1e-5,
                  'reference_passed': all(v <= 1e-5 for v in deltas.values())}
        (args.output / 'REFERENCE.json').write_text(json.dumps(report, indent=2))
        if not report['reference_passed']:
            raise ValueError('Original clean probability replay failed')
        if not args.reference_only:
            if args.root is None:
                raise ValueError('--root required for DF40 inference')
            rows = read_csv(args.root / 'manifest.csv')
            if len(rows) != 24 or len({r['image_id'] for r in rows}) != 24:
                raise ValueError('Complete 24-image manifest required')
            for row in rows:
                row['path'] = str(args.root / row['path'])
            features = encode(rows, encoder, preprocess)
            output, _ = predict(rows, features, runs)
            repeated, _ = predict(rows, encode(rows, encoder, preprocess), runs)
            delta = max(abs(a['prob_fake'] - b['prob_fake']) for a, b in zip(output, repeated))
            if delta > 1e-6:
                raise ValueError('Repeated inference differs')
            write_csv(args.output / 'df40_predictions.csv', output)
            report.update(df40_images=24, df40_scores=len(output), repeat_max_abs_difference=delta)
    if digest(weights) != backbone_hash or any(digest(r['checkpoint']) != r['sha256'] for r in runs):
        raise ValueError('Model bytes changed')
    if any(p.grad is not None for p in encoder.parameters()):
        raise ValueError('Unexpected gradient state')
    report.update(status='reference_passed_df40_pending' if args.reference_only else 'passed',
                  backbone_sha256=backbone_hash, checkpoint_sha256={r['mode']: r['sha256'] for r in runs},
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version(),
                  code_sha256=digest(Path(__file__)), preprocessing=repr(preprocess), training=False)
    (args.output / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
