rows, missing = [], []
for entry in PLAN:
    config, directory = entry['config'], entry['directory']
    for condition in EVAL_CONDITIONS:
        candidates = []
        for result_path in (directory / 'evaluations').glob(f'{condition}_*/result.json'):
            destination = result_path.parent
            if not (destination / 'COMPLETE.json').exists():
                continue
            result = json.loads(result_path.read_text())
            evaluation = json.loads((destination / 'evaluation_config.json').read_text())
            if (result['protocol_hash'] == config['protocol_hash']
                    and evaluation['degradation'] == (EVAL_DEGRADATIONS if condition != 'clean' else None)
                    and evaluation['evaluation_seed'] == EVALUATION_SEED
                    and evaluation['bins'] == CALIBRATION_BINS
                    and evaluation['engine_hash'] == ENGINE_HASH):
                candidates.append(result)
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous evaluations for {directory}/{condition}')
        if candidates:
            rows.append(candidates[0])
        else:
            missing.append({'experiment': config['experiment'], 'mode': config['mode'],
                            'seed': config['seed'], 'condition': condition})

SUMMARY_DIR = OUTPUT_ROOT / 'summaries' / digest_json({
    'protocols': [p['config']['protocol_hash'] for p in PLAN], 'conditions': EVAL_CONDITIONS,
    'degradations': EVAL_DEGRADATIONS, 'bins': CALIBRATION_BINS,
    'evaluation_seed': EVALUATION_SEED, 'families': GENERATOR_FAMILIES})[:12]
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
atomic_csv(SUMMARY_DIR / 'missing_results.csv', pd.DataFrame(missing))
print(f'{len(rows)} completed evaluations; {len(missing)} missing. Summary: {SUMMARY_DIR}')
if missing:
    display(pd.DataFrame(missing))
if rows:
    results = pd.DataFrame(rows)
    results['family'] = results.holdout.map(GENERATOR_FAMILIES).fillna('unassigned')
    atomic_csv(SUMMARY_DIR / 'all_results.csv', results)
    metrics = ['accuracy', 'f1_fake', 'auroc', 'recall_fake', 'ece', 'brier', 'log_loss']
    grouped = results.groupby(['holdout', 'k', 'mode', 'condition'], dropna=False)[metrics].agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(c) for c in grouped.columns]
    summary = grouped.reset_index()
    atomic_csv(SUMMARY_DIR / 'summary_by_k.csv', summary)
    display(summary)

    # Averaging holdout means first avoids weighting a family by number of repeats.
    family_summary = summary.copy()
    family_summary['family'] = family_summary.holdout.map(GENERATOR_FAMILIES).fillna('unassigned')
    family_summary = family_summary.groupby(['family', 'k', 'mode', 'condition']).agg(
        n_holdouts=('holdout', 'nunique'), mean_auroc=('auroc_mean', 'mean'),
        mean_accuracy=('accuracy_mean', 'mean'), mean_f1=('f1_fake_mean', 'mean')).reset_index()
    atomic_csv(SUMMARY_DIR / 'family_summary.csv', family_summary)

    keys = ['experiment', 'holdout', 'k', 'rep', 'seed', 'condition']
    base = results[results['mode'] == 'baseline'][keys + metrics]
    aug = results[results['mode'] == 'augmented'][keys + metrics]
    paired = base.merge(aug, on=keys, suffixes=('_baseline', '_augmented'), validate='one_to_one')
    for metric in metrics:
        paired[metric + '_delta'] = paired[metric + '_augmented'] - paired[metric + '_baseline']
    atomic_csv(SUMMARY_DIR / 'augmentation_deltas.csv', paired)

    for (holdout, condition), table in summary.groupby(['holdout', 'condition']):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, metric in zip(axes, ['accuracy', 'f1_fake', 'auroc']):
            for mode, subset in table.groupby('mode'):
                subset = subset.sort_values('k')
                ax.errorbar(subset.k, subset[metric+'_mean'], yerr=subset[metric+'_std'].fillna(0),
                            marker='o', capsize=3, label=mode)
            ax.set(xlabel='Number of training generators (k)', ylabel=metric, xticks=K_VALUES, ylim=(0,1))
            ax.grid(alpha=0.2)
        axes[0].legend()
        fig.suptitle(f'Held out: {holdout} | {condition} | mean +/- sample SD across available repeats/seeds')
        fig.tight_layout()
        fig.savefig(SUMMARY_DIR / f'{holdout}_{condition}_comparison.png', dpi=150)
        plt.show()

    # Heatmap is useful with one holdout now and expands as more holdouts are added.
    for (mode, condition), table in summary.groupby(['mode', 'condition']):
        pivot = table.pivot(index='holdout', columns='k', values='auroc_mean')
        fig, ax = plt.subplots(figsize=(6, max(2.2, len(pivot)*0.6)))
        values = pivot.to_numpy()
        plotted = ax.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
               yticks=range(len(pivot.index)), yticklabels=pivot.index,
               xlabel='Training generators (k)', ylabel='Held-out generator',
               title=f'{mode} | {condition} | mean test AUROC')
        for (i,j), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(j,i,f'{value:.3f}',ha='center',va='center',color='white' if value>0.6 else 'black')
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        fig.savefig(SUMMARY_DIR / f'{mode}_{condition}_auroc_heatmap.png', dpi=150)
        plt.show()
else:
    print('No matching final test results yet. Training/validation artifacts remain in each run folder.')
