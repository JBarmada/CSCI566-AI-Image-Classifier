"""Controlled FF face-adaptation experiment with a frozen CLIP image encoder."""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import clip
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

ROOT = Path('/project2/yzhao010_1531/csci566-genimage/data/df40_adaptation_v1')
TEST = Path('/project2/yzhao010_1531/csci566-genimage/data/df40_ff_v1/test')
WEIGHTS = Path.home() / '.cache/clip/ViT-L-14.pt'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def features(rows, base, model, transform):
    output = []
    for offset in range(0, len(rows), 16):
        tensors = []
        for row in rows[offset:offset + 16]:
            path = base / row['path']
            if digest(path) != row['sha256']:
                raise ValueError(f'Image hash mismatch: {path}')
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
        with torch.inference_mode():
            feature = model.encode_image(torch.stack(tensors).cuda()).float()
            output.append(F.normalize(feature, dim=-1).cpu())
    return torch.cat(output)


def choose(real_scores):
    values = [float(value) for value in real_scores]
    candidates = [0.0] + [np.nextafter(value, np.inf) for value in values]
    return float(min(v for v in candidates if sum(score >= v for score in values) / len(values) <= 0.10))


def train_head(x, y, val_x, val_y, output):
    torch.manual_seed(566)
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_epoch, state, history = -1.0, 0, None, []
    for epoch in range(1, 101):
        head.train()
        for index in torch.randperm(len(x)).split(32):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(head(x[index]).flatten(), y[index])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            scores = torch.sigmoid(head(val_x).flatten()).numpy()
        auc = float(roc_auc_score(val_y.numpy(), scores))
        history.append({'epoch': epoch, 'validation_auc': auc})
        if auc > best + 1e-12:
            best, best_epoch = auc, epoch
            state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        if epoch - best_epoch >= 15:
            break
    head.load_state_dict(state)
    with torch.no_grad():
        val_scores = torch.sigmoid(head(val_x).flatten()).numpy()
    threshold = choose(val_scores[val_y.numpy() == 0])
    torch.save({'state_dict': state, 'epoch': best_epoch, 'threshold': threshold}, output / 'head.pt')
    (output / 'selection.json').write_text(json.dumps({'epoch': best_epoch, 'validation_auc': best,
        'threshold': threshold, 'history': history, 'selection': 'best validation AUROC; patience 15'}, indent=2))
    return head, threshold


def report(rows, scores, threshold):
    real = [p for row, p in zip(rows, scores) if row['method'] == 'real']
    result = {}
    for method in ('simswap', 'fomm', 'wav2lip'):
        fake = [p for row, p in zip(rows, scores) if row['method'] == method]
        result[method] = {'real': len(real), 'fake': len(fake),
                          'fp': sum(p >= threshold for p in real),
                          'tp': sum(p >= threshold for p in fake),
                          'auroc': float(roc_auc_score([0]*len(real) + [1]*len(fake), real + fake))}
    return result


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    random.seed(566)
    np.random.seed(566)
    torch.manual_seed(566)
    torch.set_num_threads(1)
    before = digest(WEIGHTS)
    data = read(ROOT / 'manifest.csv')
    train = [r for r in data if r['split'] == 'train']
    val = [r for r in data if r['split'] == 'validation']
    if len(train) != 240 or len(val) != 120:
        raise ValueError('Unexpected adaptation split size')
    model, transform = clip.load(str(WEIGHTS), device='cuda', jit=False)
    model.eval().requires_grad_(False)
    train_x = features(train, ROOT, model, transform)
    val_x = features(val, ROOT, model, transform)
    train_y = torch.tensor([int(r['label']) for r in train], dtype=torch.float32)
    val_y = torch.tensor([int(r['label']) for r in val], dtype=torch.float32)
    head, threshold = train_head(train_x, train_y, val_x, val_y, output)
    # Selection and threshold are saved before any test features are read.
    test = read(TEST / 'manifest.csv')
    test_x = features(test, TEST, model, transform)
    with torch.no_grad():
        scores = torch.sigmoid(head(test_x).flatten()).tolist()
    if digest(WEIGHTS) != before:
        raise ValueError('Backbone changed')
    with (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test[0]) + ['prob_fake'])
        writer.writeheader()
        writer.writerows({**r, 'prob_fake': p} for r, p in zip(test, scores))
    results = {'status': 'passed', 'threshold': threshold, 'backbone_sha256': before,
               'training_manifest_sha256': digest(ROOT / 'manifest.csv'),
               'test_manifest_sha256': digest(TEST / 'manifest.csv'),
               'test_calibrated': report(test, scores, threshold), 'test_0_5': report(test, scores, .5)}
    (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
