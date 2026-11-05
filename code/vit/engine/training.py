def train_one(directory, frames, config):
    seed = config['seed']
    seed_everything(seed)
    xval, yval = deterministic_features(frames['val'])
    xtrain = ytrain = None
    if config['mode'] == 'baseline':
        xtrain, ytrain = deterministic_features(frames['train'])
    head = nn.Linear(xval.shape[1], 1).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    state = {'history': [], 'best_auc': -1.0, 'best_loss': 1e30,
             'best_state': None, 'best_epoch': None, 'stale': 0, 'epoch': 0}
    last_path = directory / 'last.pt'
    state = restore_training(last_path, head, optimizer, state, config)
    for epoch in range(state['epoch'] + 1, MAX_EPOCHS + 1):
        if state['stale'] >= PATIENCE:
            break
        report_progress(OUTPUT_ROOT, 'running', epoch=epoch, phase='training', force=True)
        seed_everything(seed + epoch * 1009)
        started = time.monotonic()
        xtrain, ytrain = epoch_features(frames, config, epoch, xtrain, ytrain)
        total_loss = train_head_epoch(head, optimizer, loss_fn, xtrain, ytrain, seed, epoch)
        probabilities = predict_features(head, xval)
        metrics = classification_metrics(yval.numpy(), probabilities)
        loss = metrics['log_loss']
        state['history'].append({'epoch': epoch, 'train_loss': total_loss / len(ytrain),
                                 'seconds': time.monotonic() - started,
                                 **{f'val_{key}': value for key, value in metrics.items()}})
        improved = better_checkpoint(metrics['auroc'], loss, state['best_auc'], state['best_loss'])
        if improved:
            state.update(best_auc=metrics['auroc'], best_loss=loss, best_epoch=epoch, stale=0,
                         best_state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        else:
            state['stale'] += 1
        state['epoch'] = epoch
        atomic_torch(last_path, {'head': {k: v.detach().cpu() for k,v in head.state_dict().items()},
                                'optimizer': optimizer.state_dict(), 'state': state,
                                'protocol_hash': config['protocol_hash']})
        atomic_csv(directory / 'training_history.csv', pd.DataFrame(state['history']))
        if STOP_AFTER_EPOCH == epoch:
            raise KeyboardInterrupt('Requested smoke-test interruption after saved epoch')
        print(f'Epoch {epoch:02d} | train loss {total_loss/len(ytrain):.4f} | '
              f'val AUROC {metrics["auroc"]:.4f} | val F1 {metrics["f1_fake"]:.4f}', flush=True)
    finish_training(directory, frames, config, head, xval, state)

def evaluation_identity(directory, config, condition):
    evaluation_config = {'protocol_hash': config['protocol_hash'],
                         'checkpoint_sha256': file_digest(directory / 'best_head.pt'), 'condition': condition,
                         'degradation': EVAL_DEGRADATIONS if condition != 'clean' else None,
                         'evaluation_seed': EVALUATION_SEED, 'bins': CALIBRATION_BINS,
                         'threshold': THRESHOLD, 'engine_hash': ENGINE_HASH}
    identity = digest_json(evaluation_config)
    return evaluation_config, identity, directory / 'evaluations' / f'{condition}_{identity[:12]}'

def evaluate_one(directory, frames, config, condition):
    best = directory / 'best_head.pt'
    evaluation_config, identity, destination = evaluation_identity(directory, config, condition)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'COMPLETE.json'
    if marker.exists():
        if json.loads(marker.read_text())['evaluation_hash'] != identity:
            raise ValueError('Evaluation identity collision.')
        for required in ['result.json', 'test_predictions.csv', 'test_metrics.json',
                         'test_calibration.csv', 'test_diagnostics.png', 'evaluation_config.json']:
            if not (destination / required).is_file():
                raise FileNotFoundError(f'Completed evaluation is missing: {destination / required}')
        return
    saved = torch.load(best, map_location='cpu', weights_only=True)
    if saved['protocol_hash'] != config['protocol_hash']:
        raise ValueError('Selected head does not match this run.')
    head = nn.Linear(saved['feature_dim'], 1).to(DEVICE)
    head.load_state_dict(saved['head_state_dict'])
    features, labels = deterministic_features(frames['test'], condition)
    probabilities = predict_features(head, features)
    atomic_json(destination / 'evaluation_config.json', evaluation_config)
    metadata = {'experiment': config['experiment'], 'holdout': config['holdout'],
                'k': config['k'], 'rep': config['rep'], 'seed': config['seed'],
                'mode': config['mode'], 'condition': condition}
    metrics = export_predictions(destination, 'test', frames['test'], probabilities, metadata)
    atomic_json(destination / 'result.json', {**metadata, **metrics,
                 'protocol_hash': config['protocol_hash'], 'evaluation_hash': identity,
                 'family': GENERATOR_FAMILIES.get(config['holdout'], 'unassigned'),
                 'run_dir': str(directory), 'evaluation_dir': str(destination)})
    atomic_json(marker, {'evaluation_hash': identity, 'completed_utc': utc_now()})
    print(f'Test {condition}: accuracy={metrics["accuracy"]:.4f}, '
          f'F1={metrics["f1_fake"]:.4f}, AUROC={metrics["auroc"]:.4f}', flush=True)
