"""Score the frozen candidate on CPU after recording reference mismatches."""
import json
from pathlib import Path
import sys

import clip
from PIL import Image
import torch
from torchvision.models import resnet50

ROOT = Path('/project2/yzhao010_1531/csci566-genimage')
sys.path.insert(0, str(ROOT / 'results/df40_ff_eval_v1'))
import smoke_inference as vit
import evaluate_shared as shared

DATA = ROOT / 'data/df40_fresh_candidate_review_v3_20261123'
ADAPT = ROOT / 'results/df40_adaptation_v1/adapt_results_14482534/head.pt'
MANIFEST_HASH = '1f18b71d5d4ebd59ff184d1db06cb33f82f99f034274f6aaaa51b72724afae51'
ADAPT_HASH = '8d0fdf3893fc7362801e8801c4fae9fe87e60f069a43803af3aade394e079af2'
WEIGHTS_HASH = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def load_rows():
    manifest = DATA / 'manifest.csv'
    if vit.digest(manifest) != MANIFEST_HASH:
        raise ValueError('Candidate manifest hash differs from frozen protocol')
    rows = vit.read_csv(manifest)
    if len(rows) != 189 or len({row['sha256'] for row in rows}) != 189:
        raise ValueError('Unexpected candidate count')
    for row in rows:
        row['image_id'] = row['sha256']
        row['path'] = str(DATA / row['path'])
    return rows


def cnn_scores(mode, rows):
    folder = ROOT / ('results/cnn_benchmark/notebook/runs/'
                     f'holdout_ADM_k7_rep0__{mode}__seed17')
    marker = json.loads((folder / 'TRAINING_COMPLETE.json').read_text())
    checkpoint = folder / 'best.pt'
    if vit.digest(checkpoint) != marker['checkpoint_sha256']:
        raise ValueError('CNN checkpoint hash mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['protocol_hash'] != marker['protocol_hash']:
        raise ValueError('CNN protocol mismatch')
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    transform = shared.cnn_transform()
    scores = []
    for start in range(0, len(rows), 16):
        tensors = []
        for row in rows[start:start + 16]:
            if vit.digest(Path(row['path'])) != row['sha256']:
                raise ValueError('Candidate image hash mismatch')
            with Image.open(row['path']) as image:
                tensors.append(transform(image.convert('RGB')))
        scores.extend(torch.sigmoid(model(torch.stack(tensors)).flatten()).tolist())
    return scores


def clip_scores(rows):
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self.cpu()
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.cpu()
    weights = Path.home() / '.cache/clip/ViT-L-14.pt'
    if vit.digest(weights) != WEIGHTS_HASH or vit.digest(ADAPT) != ADAPT_HASH:
        raise ValueError('CLIP backbone or adapted head hash mismatch')
    runs = vit.load_runs()
    encoder, transform = clip.load(str(weights), device='cpu', jit=False)
    encoder.eval().requires_grad_(False)
    features = vit.encode(rows, encoder, transform)
    predicted, _ = vit.predict(rows, features, runs)
    saved = torch.load(ADAPT, map_location='cpu', weights_only=False)
    if abs(float(saved['threshold']) - 0.5676835775375367) > 1e-12:
        raise ValueError('Adapted threshold mismatch')
    head = torch.nn.Linear(saved['state_dict']['weight'].shape[1], 1)
    head.load_state_dict(saved['state_dict'])
    head.eval().requires_grad_(False)
    adapted = torch.sigmoid(head(features).flatten()).tolist()
    return predicted, adapted


def main():
    out, reference = Path(sys.argv[1]), Path(sys.argv[2])
    check = json.loads(reference.read_text())
    if check['status'] != 'mismatch' or check['device'] != 'cpu':
        raise ValueError('Expected recorded CPU reference mismatch')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    rows, output = load_rows(), []
    with torch.inference_mode():
        for mode in ('baseline', 'augmented'):
            scores = cnn_scores(mode, rows)
            output.extend({**r, 'model': 'cnn', 'mode': mode, 'prob_fake': p}
                          for r, p in zip(rows, scores))
            print(f'CNN {mode}: {len(scores)} scores', flush=True)
        predictions, adapted = clip_scores(rows)
        output.extend({**row, 'model': 'vit'} for row in predictions)
        output.extend({**row, 'model': 'vit', 'mode': 'face_adapted', 'prob_fake': p}
                      for row, p in zip(rows, adapted))
    if len(output) != 189 * 5:
        raise ValueError('Incomplete prediction count')
    vit.write_csv(out / 'predictions.csv', output)
    report = {'status': 'exploratory_cpu_reference_mismatch', 'training': False,
              'manifest_sha256': MANIFEST_HASH, 'samples': len(rows), 'scores': len(output),
              'reference_report': str(reference), 'reference_sha256': vit.digest(reference)}
    (out / 'STATUS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
