session_rows = []
active_runs = 0
selected_plan = PLAN
for run_index, entry in enumerate(selected_plan, 1):
    config, directory = entry['config'], entry['directory']
    name = config['experiment']
    directory.mkdir(parents=True, exist_ok=True)
    needs_training = not (directory / 'TRAINING_COMPLETE.json').exists()
    needs_evaluation = RUN_FINAL_TEST and not needs_training and any(
        not (evaluation_identity(directory, config, c)[2] / 'COMPLETE.json').exists()
        for c in EVAL_CONDITIONS)
    needs_work = (RUN_TRAINING and needs_training) or needs_evaluation
    if needs_work and MAX_RUNS_THIS_SESSION is not None and active_runs >= MAX_RUNS_THIS_SESSION:
        session_rows.append({'experiment': name, 'mode': config['mode'], 'seed': config['seed'],
                             'status': 'deferred by session limit'})
        continue
    if needs_work:
        active_runs += 1
    print(f'\n{name} | {config["mode"]} | seed {config["seed"]}', flush=True)
    report_progress(OUTPUT_ROOT, 'running', experiment=name, mode=config['mode'],
                    run_index=run_index, total=len(PLAN), epoch=0)
    try:
        with run_lock(directory):
            config = resolve_run_config(directory, config)
            entry['config'] = config
            atomic_json(directory / 'last_resume_check.json', {
                'checked_utc': utc_now(), 'protocol_hash': config['protocol_hash'],
                'canonical_hash': digest_json({k: v for k, v in config.items() if k != 'protocol_hash'}),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
            config_path = directory / 'config.json'
            if config_path.exists():
                previous = json.loads(config_path.read_text())
                if previous['protocol_hash'] != config['protocol_hash']:
                    raise ValueError(f'Protocol changed for {directory}. Choose a new STUDY_NAME.')
            else:
                atomic_json(config_path, {**config, 'created_utc': utc_now(),
                                          'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None})
            for split in ['train', 'val', 'test']:
                source = MANIFEST_ROOT / name / f'{split}.csv'
                destination = directory / f'input_{split}.csv'
                if file_digest(source) != config['manifest_sha256'][split]:
                    raise ValueError('Input manifest changed since preflight.')
                if destination.exists():
                    if file_digest(destination) != config['manifest_sha256'][split]:
                        raise ValueError(f'Archived manifest changed: {destination}')
                else:
                    destination.write_bytes(source.read_bytes())
            complete = directory / 'TRAINING_COMPLETE.json'
            if complete.exists():
                marker = json.loads(complete.read_text())
                if (marker['protocol_hash'] != config['protocol_hash'] or
                        file_digest(directory / 'best_head.pt') != marker['checkpoint_sha256']):
                    raise ValueError('Completed checkpoint does not match its marker.')
                for required in ['training_history.csv', 'val_predictions.csv', 'val_metrics.json']:
                    if not (directory / required).is_file():
                        raise FileNotFoundError(f'Incomplete completed run: {required}')
                print('Verified completed training; skipping.', flush=True)
            elif RUN_TRAINING:
                train_one(directory, FRAMES[name], config)
            else:
                print('Training incomplete; skipped because RUN_TRAINING=False.')
                session_rows.append({'experiment': name, 'mode': config['mode'], 'seed': config['seed'],
                                     'status': 'training incomplete'})
                continue
            if RUN_FINAL_TEST:
                for condition in EVAL_CONDITIONS:
                    evaluate_one(directory, FRAMES[name], config, condition)
            session_rows.append({'experiment': name, 'mode': config['mode'], 'seed': config['seed'],
                                 'status': 'complete' if RUN_FINAL_TEST else 'trained; test skipped'})
            report_progress(OUTPUT_ROOT, 'running', completed=run_index, total=len(PLAN))
    except Exception as error:
        report_progress(OUTPUT_ROOT, 'failed', error=str(error))
        failure_dir = OUTPUT_ROOT / 'failures'
        failure_dir.mkdir(exist_ok=True)
        atomic_json(failure_dir / f'{uuid.uuid4().hex}.json', {
            'run_dir': str(directory), 'time': utc_now(), 'error': repr(error),
            'traceback': traceback.format_exc()})
        session_rows.append({'experiment': name, 'mode': config['mode'], 'seed': config['seed'],
                             'status': f'failed: {error}'})
        if not CONTINUE_ON_ERROR:
            raise
        print('FAILED:', error, flush=True)
    finally:
        if DEVICE.type == 'cuda':
            # Cleanup must not hide the original exception/traceback.
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                print(f'CUDA cleanup warning: {cleanup_error}', flush=True)
display(pd.DataFrame(session_rows))
