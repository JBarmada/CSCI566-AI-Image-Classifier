def better_checkpoint(auc, loss, best_auc, best_loss):
    return auc > best_auc + AUROC_TOLERANCE or (
        abs(auc - best_auc) <= AUROC_TOLERANCE and loss < best_loss)


def epoch_features(frames, config, epoch, xtrain, ytrain):
    if config['mode'] != 'augmented':
        return xtrain, ytrain
    xs, ys = [], []
    loader = image_loader(frames['train'], ImageTransform(PREPROCESS, augmented=True))
    for index, (images, labels) in enumerate(tqdm(loader, desc=f'Augmented features epoch {epoch}', leave=False), 1):
        with torch.no_grad():
            xs.append(encode_batch(images).cpu())
        ys.append(labels.float())
        report_progress(OUTPUT_ROOT, 'running', epoch=epoch, batch=index,
                        batches=len(loader), phase='augmented features')
    return torch.cat(xs), torch.cat(ys)


def train_head_epoch(head, optimizer, loss_fn, xtrain, ytrain, seed, epoch):
    head.train()
    loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=HEAD_BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))
    total_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(features.to(DEVICE)).squeeze(-1), labels.to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss


def restore_training(last_path, head, optimizer, state, config):
    if last_path.exists():
        if not RESUME:
            raise FileExistsError(f'{last_path} exists; enable RESUME or use another STUDY_NAME.')
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        if saved['protocol_hash'] != config['protocol_hash']:
            raise ValueError('Checkpoint protocol mismatch.')
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        # AdamW optimizer tensors follow parameters to the active device.
        for opt_state in optimizer.state.values():
            for name, value in opt_state.items():
                if torch.is_tensor(value) and name != 'step':
                    opt_state[name] = value.to(DEVICE)
        state = saved['state']
        print('Resuming after epoch', state['epoch'])
    return state

def finish_training(directory, frames, config, head, xval, state):
    seed = config['seed']
    if state['best_state'] is None:
        raise RuntimeError('Training produced no selected checkpoint.')
    head.load_state_dict(state['best_state'])
    atomic_torch(directory / 'best_head.pt', {'head_state_dict': state['best_state'],
                 'feature_dim': int(xval.shape[1]), 'best_epoch': state['best_epoch'],
                 'protocol_hash': config['protocol_hash']})
    export_predictions(directory, 'val', frames['val'], predict_features(head, xval),
                       {'experiment': config['experiment'], 'mode': config['mode'], 'seed': seed})
    history = pd.DataFrame(state['history'])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.epoch, history.train_loss, label='Train')
    axes[0].plot(history.epoch, history.val_log_loss, label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='BCE loss')
    axes[0].legend()
    axes[1].plot(history.epoch, history.val_auroc)
    axes[1].set(xlabel='Epoch', ylabel='Validation AUROC', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(directory / 'training_curves.png', dpi=150)
    plt.close(fig)
    # Written last: a completed marker means all required outputs exist.
    atomic_json(directory / 'TRAINING_COMPLETE.json', {
        'protocol_hash': config['protocol_hash'], 'best_epoch': state['best_epoch'],
        'completed_utc': utc_now(), 'checkpoint_sha256': file_digest(directory / 'best_head.pt')})
