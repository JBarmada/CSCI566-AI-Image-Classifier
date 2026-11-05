import sys
sys.path.insert(0, str(VIT_ENGINE_ROOT.parent))
from benchmark_runtime import validate_benchmark, prepare_smoke, report_progress
DATASET_FINGERPRINT = validate_benchmark(DATA_ROOT)
if SMOKE_TEST:
    MANIFEST_ROOT = prepare_smoke(MANIFEST_ROOT, OUTPUT_ROOT, HOLDOUTS[0])
    HOLDOUTS, K_VALUES, REPEATS = [HOLDOUTS[0]], [1], [0]
    EXPECTED_COUNTS = {'train': {0: 32, 1: 32}, 'val': {0: 16, 1: 16}, 'test': {0: 16, 1: 16}}
    MAX_EPOCHS = 2
validate_settings()
ENVIRONMENT = package_versions()
ENGINE_HASH = code_fingerprint()
experiment_names = [f'holdout_{g}_k{k}_rep{r}'
                    for g, k, r in itertools.product(HOLDOUTS, K_VALUES, REPEATS)]
if EXPERIMENT_ALLOWLIST:
    unknown = set(EXPERIMENT_ALLOWLIST) - set(experiment_names)
    if unknown:
        raise ValueError(f'Allowlisted experiments outside selected plan: {unknown}')
    experiment_names = [n for n in experiment_names if n in EXPERIMENT_ALLOWLIST]

FRAMES, EXPERIMENT_INFO = {}, {}
for name in tqdm(experiment_names, desc='Verify manifests/images'):
    FRAMES[name], EXPERIMENT_INFO[name] = load_experiment(name)
ensure_encoder()
PLAN = []
for name, seed, mode in itertools.product(experiment_names, SEEDS, TRAINING_MODES):
    info = EXPERIMENT_INFO[name]
    config = {**info, 'seed': seed, 'mode': mode, 'model': MODEL_NAME,
              'backbone_frozen': True, 'features_l2_normalized': True,
              'label_mapping': {'real': 0, 'fake': 1}, 'threshold': THRESHOLD,
              'augmentation': AUGMENTATION if mode == 'augmented' else None,
              'image_batch_size': IMAGE_BATCH_SIZE, 'head_batch_size': HEAD_BATCH_SIZE,
              'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
              'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
              'selection': 'validation AUROC; ties within 1e-12 use validation BCE',
              'auroc_tolerance': AUROC_TOLERANCE,
              'content_hashes_verified': VERIFY_IMAGE_HASHES, 'num_workers': NUM_WORKERS,
              'expected_counts': EXPECTED_COUNTS,
              'preprocessing': canonical_protocol({'preprocessing': repr(PREPROCESS)})['preprocessing'],
              'engine_hash': ENGINE_HASH, 'environment': ENVIRONMENT,
              'device_type': DEVICE.type}
    config['protocol_hash'] = digest_json(config)
    directory = OUTPUT_ROOT / 'runs' / f'{name}__{mode}__seed{seed}'
    config = resolve_run_config(directory, config)
    PLAN.append({'config': config, 'directory': directory})
plan_table = pd.DataFrame([{
    'experiment': p['config']['experiment'], 'mode': p['config']['mode'], 'seed': p['config']['seed'],
    'train_generators': ', '.join(p['config']['train_generators']),
    'training_complete': (p['directory'] / 'TRAINING_COMPLETE.json').exists(),
} for p in PLAN])
display(plan_table)
print(f'{len(PLAN)} selected training runs; output root: {OUTPUT_ROOT}')
print('Test evaluation enabled:', RUN_FINAL_TEST, '| conditions:', EVAL_CONDITIONS)
print('Content hashes verified:', VERIFY_IMAGE_HASHES)
