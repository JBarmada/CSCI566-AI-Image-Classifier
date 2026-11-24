"""Evaluate preselected, unscored face candidates with existing frozen heads."""
import csv
import json
from pathlib import Path
import sys

import clip
import torch

PROJECT = Path('/project2/yzhao010_1531/csci566-genimage')
EVAL = PROJECT / 'results/df40_ff_eval_v1'
ADAPT = PROJECT / 'results/df40_adaptation_v1'
DATA = PROJECT / 'data/df40_fresh_candidate_review_v2_20261123'
HEAD = ADAPT / 'adapt_results_14482534/head.pt'
MANIFEST_HASH = '1ccc573a2add842afafc23dbed69ebea337e94a2c4893725e7e1b6a0d7c4b7b4'
HEAD_HASH = '8d0fdf3893fc7362801e8801c4fae9fe87e60f069a43803af3aade394e079af2'
BACKBONE_HASH = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
METHODS = ('real', 'simswap', 'fomm', 'wav2lip', 'inswap', 'pirender')
sys.path[:0] = [str(EVAL), str(ADAPT)]
import evaluate_shared as ev
import train_clip_head as exp


def load_rows():
    path = DATA / 'manifest.csv'
    if ev.vit.digest(path) != MANIFEST_HASH:
        raise ValueError('Candidate manifest hash differs from frozen protocol')
    rows = ev.vit.read_csv(path)
    counts = {method: sum(r['method'] == method for r in rows) for method in METHODS}
    if counts != {'real': 58, **{method: 29 for method in METHODS[1:]}}:
        raise ValueError(f'Unexpected method counts: {counts}')
    if len(rows) != 203 or len({r['sha256'] for r in rows}) != 203:
        raise ValueError('Duplicate or missing candidate rows')
    for row in rows:
        row['image_id'] = row['sha256']
        row['path'] = str(DATA / row['path'])
    return rows


def adapted(rows, smoke):
    if exp.digest(HEAD) != HEAD_HASH or exp.digest(exp.WEIGHTS) != BACKBONE_HASH:
        raise ValueError('Adapted model hash mismatch')
    saved = torch.load(HEAD, map_location='cpu', weights_only=False)
    if abs(float(saved['threshold']) - 0.5676835775375367) > 1e-12:
        raise ValueError('Adapted threshold differs from protocol')
    head = torch.nn.Linear(saved['state_dict']['weight'].shape[1], 1)
    head.load_state_dict(saved['state_dict'])
    head.eval().requires_grad_(False)
    encoder, transform = clip.load(str(exp.WEIGHTS), device='cuda', jit=False)
    encoder.eval().requires_grad_(False)
    smoke_rows = [rows[i] for i in smoke]
    first = exp.features(smoke_rows, Path('/'), encoder, transform)
    second = exp.features(smoke_rows, Path('/'), encoder, transform)
    with torch.inference_mode():
        difference = float(torch.max(torch.abs(torch.sigmoid(head(first)) - torch.sigmoid(head(second)))))
        scores = torch.sigmoid(head(exp.features(rows, Path('/'), encoder, transform)).flatten()).tolist()
    if difference > 1e-6 or exp.digest(HEAD) != HEAD_HASH or exp.digest(exp.WEIGHTS) != BACKBONE_HASH:
        raise ValueError('Adapted replay or model hash check failed')
    return scores, {'repeat_max_difference': difference, 'checkpoint_sha256': HEAD_HASH,
                    'backbone_sha256': BACKBONE_HASH}


def main():
    out = Path(sys.argv[1])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    torch.set_num_threads(1)
    rows = load_rows()
    smoke = [i for method in METHODS for i in [j for j, row in enumerate(rows) if row['method'] == method][:3]]
    out.mkdir(parents=True, exist_ok=False)
    report = {'training': False, 'manifest_sha256': MANIFEST_HASH,
              'gpu': torch.cuda.get_device_name(0), 'models': {}}
    predictions = []
    with torch.inference_mode():
        for mode in ('baseline', 'augmented'):
            scores, checks = ev.cnn_run(mode, rows, smoke)
            report['models']['cnn_' + mode] = checks
            predictions.extend({**r, 'model': 'cnn', 'mode': mode, 'prob_fake': p} for r, p in zip(rows, scores))
        scores, report['models']['vit'] = ev.vit_run(rows, smoke)
        predictions.extend({**r, 'model': 'vit'} for r in scores)
        scores, report['models']['adapted'] = adapted(rows, smoke)
        predictions.extend({**r, 'model': 'vit', 'mode': 'face_adapted', 'prob_fake': p}
                           for r, p in zip(rows, scores))
    if len(predictions) != 203 * 5:
        raise ValueError('Incomplete model predictions')
    ev.vit.write_csv(out / 'predictions.csv', predictions)
    report.update(status='passed', samples=len(rows), scores=len(predictions))
    (out / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
