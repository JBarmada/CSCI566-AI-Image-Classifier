def reliability_bins(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    index = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        selected = index == b
        count = int(selected.sum())
        rows.append({'bin': b, 'lower': b / bins, 'upper': (b + 1) / bins, 'count': count,
                     'mean_prob_fake': float(probabilities[selected].mean()) if count else None,
                     'observed_fake_fraction': float(labels[selected].mean()) if count else None})
    return pd.DataFrame(rows)

def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or probabilities.shape != labels.shape or len(labels) == 0
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Expected finite binary labels and aligned probabilities in [0,1].')
    labels = labels.astype(int)
    predicted = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    table = reliability_bins(labels, probabilities, CALIBRATION_BINS)
    occupied = table[table['count'] > 0]
    ece = ((occupied['mean_prob_fake'] - occupied['observed_fake_fraction']).abs()
           * occupied['count']).sum() / len(labels)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {'n': len(labels), 'threshold': THRESHOLD,
            'accuracy': float(accuracy_score(labels, predicted)),
            'precision_fake': float(precision_score(labels, predicted, zero_division=0)),
            'recall_fake': float(recall_score(labels, predicted, zero_division=0)),
            'f1_fake': float(f1_score(labels, predicted, zero_division=0)),
            'auroc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            'ece': float(ece), 'brier': float(np.mean((probabilities - labels) ** 2)),
            'log_loss': float(-np.mean(labels * np.log(clipped) + (1-labels) * np.log1p(-clipped))),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

@torch.no_grad()
def predict_features(head, features):
    head.eval()
    return torch.cat([torch.sigmoid(head(batch.to(DEVICE)).squeeze(-1)).cpu()
                      for batch in features.split(HEAD_BATCH_SIZE)]).numpy()

def export_predictions(directory, stem, frame, probabilities, metadata):
    result = frame.drop(columns=['resolved_path']).copy()
    result['image_id'] = result.sha256
    result['prob_fake'] = probabilities
    result['pred_label'] = (probabilities >= THRESHOLD).astype(int)
    for key, value in metadata.items():
        result[key] = value
    result['threshold'] = THRESHOLD
    atomic_csv(directory / f'{stem}_predictions.csv', result)
    metrics = classification_metrics(result.label, probabilities)
    atomic_json(directory / f'{stem}_metrics.json', metrics)
    table = reliability_bins(result.label, probabilities, CALIBRATION_BINS)
    atomic_csv(directory / f'{stem}_calibration.csv', table)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    occupied = table[table['count'] > 0]
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].plot(occupied.mean_prob_fake, occupied.observed_fake_fraction, 'o-')
    axes[0].set(xlabel='Mean predicted fake probability', ylabel='Observed fake fraction',
                xlim=(0, 1), ylim=(0, 1), title=f'{stem}: ECE={metrics["ece"]:.3f}')
    matrix = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    axes[1].imshow(matrix, cmap='Blues')
    for (i, j), count in np.ndenumerate(matrix):
        axes[1].text(j, i, str(count), ha='center', va='center',
                     color='white' if count > matrix.max()/2 else 'black')
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'], xlabel='Predicted label', ylabel='True label',
                title=f'Accuracy={metrics["accuracy"]:.3f}, AUROC={metrics["auroc"]:.3f}')
    fig.tight_layout()
    fig.savefig(directory / f'{stem}_diagnostics.png', dpi=150)
    plt.close(fig)
    return metrics
