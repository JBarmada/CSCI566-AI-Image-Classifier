"""Replay the saved adaptation head and recover metrics after a write failure."""
import json
import sys
from pathlib import Path

import clip
import torch
import train_clip_head as exp


def main():
    original = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = torch.load(original / 'head.pt', map_location='cpu', weights_only=False)
    selection = json.loads((original / 'selection.json').read_text())
    if saved['epoch'] != selection['epoch'] or float(saved['threshold']) != selection['threshold']:
        raise ValueError('Saved head and selection do not agree')
    head = torch.nn.Linear(saved['state_dict']['weight'].shape[1], 1)
    head.load_state_dict(saved['state_dict'])
    head.eval().requires_grad_(False)
    before = exp.digest(exp.WEIGHTS)
    model, transform = clip.load(str(exp.WEIGHTS), device='cuda', jit=False)
    model.eval().requires_grad_(False)
    validation = [r for r in exp.read(exp.ROOT / 'manifest.csv') if r['split'] == 'validation']
    val_features = exp.features(validation, exp.ROOT, model, transform)
    with torch.inference_mode():
        val_scores = torch.sigmoid(head(val_features).flatten()).numpy()
    labels = [int(r['label']) for r in validation]
    auc = float(exp.roc_auc_score(labels, val_scores))
    threshold = exp.choose(val_scores[[label == 0 for label in labels]])
    if abs(auc - selection['validation_auc']) > 1e-9 or threshold != selection['threshold']:
        raise ValueError('Validation selection replay failed')
    rows = exp.read(exp.TEST / 'manifest.csv')
    old = exp.read(original / 'test_predictions.csv')
    if len(rows) != 280 or [r['sha256'] for r in old] != [r['sha256'] for r in rows]:
        raise ValueError('Saved test predictions do not match the test manifest')
    features = exp.features(rows, exp.TEST, model, transform)
    with torch.inference_mode():
        scores = torch.sigmoid(head(features).flatten()).tolist()
    difference = max(abs(a - float(b['prob_fake'])) for a, b in zip(scores, old))
    if difference > 1e-6 or exp.digest(exp.WEIGHTS) != before:
        raise ValueError(f'Replay failed: {difference}')
    result = {'status': 'recovered_after_summary_write_failure', 'original_job': '14482534',
              'max_replay_difference': difference, 'threshold': threshold,
              'validation_auc_replayed': auc, 'validation_real_fpr': sum(
                  float(score) >= threshold for score, label in zip(val_scores, labels) if label == 0) / labels.count(0),
              'training_manifest_sha256': exp.digest(exp.ROOT / 'manifest.csv'),
              'head_sha256': exp.digest(original / 'head.pt'), 'backbone_sha256': before,
              'selection_sha256': exp.digest(original / 'selection.json'),
              'test_manifest_sha256': exp.digest(exp.TEST / 'manifest.csv'),
              'test_calibrated': exp.report(rows, scores, threshold),
              'test_0_5': exp.report(rows, scores, .5)}
    (out / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
