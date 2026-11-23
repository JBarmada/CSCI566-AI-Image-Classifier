"""Independently replay selected objective heads on saved CLIP features."""
import json
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score
import torch

import run
from objectives import Adapter


def main():
    original = Path(sys.argv[1])
    output = Path(sys.argv[2])
    config = json.loads((run.RUN / 'config.json').read_text())
    validation = run.load_features('val', config)
    test = run.load_features('test', config)
    checks = {}
    for name in ('baseline', 'adversarial', 'contrastive'):
        checkpoint = torch.load(original / (name + '_head.pt'), map_location='cpu', weights_only=True)
        selection = json.loads((original / (name + '_selection.json')).read_text())
        if checkpoint['epoch'] != selection['epoch']:
            raise ValueError('Selection epoch mismatch')
        model = Adapter(768, 7)
        model.load_state_dict(checkpoint['state_dict'])
        validation_auc = float(roc_auc_score(validation[2].numpy(), run.scores(model, validation[1])))
        previous = {r['sha256']: float(r['prob_fake']) for r in read_predictions(original, name)}
        values = run.scores(model, test[1])
        difference = max(abs(float(value) - previous[row['sha256']]) for row, value in zip(test[0], values))
        if abs(validation_auc - selection['validation_auc']) > 1e-9 or difference > 1e-7:
            raise ValueError(f'Replay failed for {name}: {difference}')
        checks[name] = {'validation_auc': validation_auc, 'test_max_abs_difference': difference,
                        'checkpoint_sha256': run.digest(original / (name + '_head.pt'))}
    output.write_text(json.dumps({'status': 'passed', 'checks': checks}, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


def read_predictions(directory, name):
    return run.read(directory / (name + '_predictions.csv'))


if __name__ == '__main__':
    main()
