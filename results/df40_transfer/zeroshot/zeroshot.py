"""Score fixed CLIP text prompts on a pre-verified shared manifest."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/project2/yzhao010_1531/csci566-genimage/data/df40_shared_v1')
WEIGHTS = Path.home() / '.cache/clip/ViT-L-14.pt'
CONFIG = Path(__file__).parent / 'prompts.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read_rows():
    with (ROOT / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 500 or len({r['sha256'] for r in rows}) != 500:
        raise ValueError('Unexpected manifest')
    return rows


def image_tensor(row, preprocess):
    path = ROOT / row['path']
    if digest(path) != row['sha256']:
        raise ValueError('Image hash mismatch: ' + str(path))
    with Image.open(path) as image:
        return preprocess(image.convert('RGB'))


def main():
    if not torch.cuda.is_available():
        raise ValueError('GPU allocation required')
    torch.set_num_threads(1)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    before = digest(WEIGHTS)
    rows = read_rows()
    model, preprocess = clip.load(str(WEIGHTS), device='cuda', jit=False)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        tokens = clip.tokenize([config['real_prompt'], config['fake_prompt']]).cuda()
        texts = F.normalize(model.encode_text(tokens).float(), dim=-1)
        scale = model.logit_scale.exp().float()
        output = []
        for start in range(0, len(rows), 16):
            part = rows[start:start + 16]
            batch = torch.stack([image_tensor(r, preprocess) for r in part]).cuda()
            features = F.normalize(model.encode_image(batch).float(), dim=-1)
            scores = (scale * features @ texts.T).softmax(dim=-1)[:, 1].cpu().tolist()
            if not all(0 <= score <= 1 for score in scores):
                raise ValueError('Invalid score')
            output.extend({**r, 'prob_fake_prompt': s} for r, s in zip(part, scores))
    if digest(WEIGHTS) != before:
        raise ValueError('Backbone changed')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    report = {'status': 'passed', 'images': len(output), 'training': False,
              'backbone_sha256': before, 'manifest_sha256': digest(ROOT / 'manifest.csv'),
              'prompt_sha256': digest(CONFIG), 'prompts': config,
              'gpu': torch.cuda.get_device_name(0), 'logit_scale': scale.item(),
              'preprocessing': repr(preprocess)}
    (out / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
