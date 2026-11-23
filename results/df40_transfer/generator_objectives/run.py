"""Compare three feature objectives using verified cached CLIP features."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from objectives import Adapter, loss_for

ROOT = Path('/project2/yzhao010_1531/csci566-genimage')
RUN = ROOT / 'results/vit_benchmark/ADM/runs/holdout_ADM_k7_rep0__baseline__seed17'
CACHE = ROOT / 'features/vit_benchmark/ADM'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load_features(name, config):
    manifest = RUN / ('input_' + name + '.csv')
    rows = read(manifest)
    identity = {'images': [r['sha256'] for r in rows], 'model': config['model'],
                'preprocessing': config['preprocessing'], 'l2_normalized': True,
                'condition': 'clean', 'degradation': None, 'evaluation_seed': None,
                'engine': config['engine_hash'], 'environment': config['environment'],
                'device_type': config['device_type'], 'image_batch_size': config['image_batch_size']}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    file = CACHE / (key + '.pt')
    cached = torch.load(file, map_location='cpu', weights_only=True)
    labels = torch.tensor([int(r['label']) for r in rows], dtype=torch.float32)
    if cached['key'] != key or not torch.equal(cached['labels'], labels):
        raise ValueError('Cache identity or labels differ')
    if cached['features'].shape != (len(rows), 768) or not torch.isfinite(cached['features']).all():
        raise ValueError('Unexpected feature cache')
    return rows, cached['features'].float(), labels, {'key': key, 'cache_sha256': digest(file),
                                                      'manifest_sha256': digest(manifest)}


def replay_linear(features, rows, path, head):
    previous = {r['sha256']: float(r['prob_fake']) for r in read(path)}
    with torch.no_grad():
        scores = torch.sigmoid(head(features).flatten()).tolist()
    delta = max(abs(score - previous[row['sha256']]) for score, row in zip(scores, rows))
    if delta > 1e-5:
        raise ValueError(f'Original head replay failed: {delta}')
    return delta


def scores(model, x):
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
        return torch.sigmoid(logits).numpy()


def train(mode, train_data, valid_data, domain_count, output):
    torch.manual_seed(17)
    x, y, domains = train_data
    vx, vy = valid_data
    model = Adapter(x.shape[1], domain_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(17)
    loader = DataLoader(TensorDataset(x, y, domains), batch_size=128, shuffle=True, generator=generator)
    best, best_epoch, state, history = -1.0, 0, None, []
    for epoch in range(1, 21):
        model.train()
        for batch_x, batch_y, batch_d in loader:
            optimizer.zero_grad()
            loss = loss_for(model, batch_x, batch_y, batch_d, mode)
            loss.backward()
            optimizer.step()
        auc = float(roc_auc_score(vy.numpy(), scores(model, vx)))
        history.append({'epoch': epoch, 'validation_auc': auc})
        if auc > best + 1e-12:
            best, best_epoch = auc, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 3:
            break
    model.load_state_dict(state)
    torch.save({'state_dict': state, 'epoch': best_epoch}, output / (mode + '_head.pt'))
    (output / (mode + '_selection.json')).write_text(json.dumps({'epoch': best_epoch,
        'validation_auc': best, 'history': history}, indent=2))
    return model


def evaluate(model, x, labels, rows, name, output):
    probability = scores(model, x)
    result = {'objective': name, 'n': len(rows), 'accuracy': float(np.mean((probability >= .5) == labels.numpy())),
              'auroc': float(roc_auc_score(labels.numpy(), probability)),
              'real_fp': int(np.sum((probability >= .5) & (labels.numpy() == 0))),
              'fake_tp': int(np.sum((probability >= .5) & (labels.numpy() == 1)))}
    with (output / (name + '_predictions.csv')).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sha256', 'label', 'prob_fake'])
        writer.writeheader()
        writer.writerows({'sha256': r['sha256'], 'label': r['label'], 'prob_fake': p}
                         for r, p in zip(rows, probability))
    return result


def main():
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((RUN / 'config.json').read_text())
    marker = json.loads((RUN / 'TRAINING_COMPLETE.json').read_text())
    if config['protocol_hash'] != marker['protocol_hash'] or digest(RUN / 'best_head.pt') != marker['checkpoint_sha256']:
        raise ValueError('Original checkpoint identity mismatch')
    loaded = {name: load_features(name, config) for name in ('train', 'val', 'test')}
    for name, (rows, _, _, _) in loaded.items():
        if len(rows) != (2800 if name == 'train' else 560):
            raise ValueError(f'Unexpected {name} size')
    checkpoint = torch.load(RUN / 'best_head.pt', map_location='cpu', weights_only=True)
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(checkpoint['head_state_dict'])
    head.eval()
    tests = list((RUN / 'evaluations').glob('clean_*/test_predictions.csv'))
    if len(tests) != 1:
        raise ValueError('Expected one original test result')
    replay = {name: replay_linear(loaded[name][1], loaded[name][0], path, head) for name, path in
              [('val', RUN / 'val_predictions.csv'), ('test', tests[0])]}
    train_rows, train_x, train_y, _ = loaded['train']
    names = sorted({r['source_generator'] for r in train_rows if r['label'] == '1'})
    if len(names) != 7 or 'ADM' in names:
        raise ValueError('Unexpected training generator domains')
    domain = torch.tensor([names.index(r['source_generator']) if r['label'] == '1' else 0
                           for r in train_rows], dtype=torch.long)
    models = {name: train(name, (train_x, train_y, domain), (loaded['val'][1], loaded['val'][2]),
                          len(names), out) for name in ('baseline', 'adversarial', 'contrastive')}
    result = [evaluate(model, loaded['test'][1], loaded['test'][2], loaded['test'][0], name, out)
              for name, model in models.items()]
    record = {'status': 'passed', 'replay_max_difference': replay, 'domains': names,
              'cache': {name: value[3] for name, value in loaded.items()}, 'results': result,
              'limit': 'One previously inspected ADM holdout; frozen features and one seed.'}
    (out / 'RESULTS.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
