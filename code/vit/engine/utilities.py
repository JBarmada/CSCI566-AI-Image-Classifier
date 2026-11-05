ENGINE_VERSION = '2.0.0-benchmark'

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def canonical_protocol(value):
    # Transform repr includes a function address that changes on every kernel restart.
    # Only remove that address; retain the function name and every transform setting.
    if isinstance(value, dict):
        return {key: (re.sub(r'(<function [^>]+?) at 0x[0-9a-fA-F]+>', r'\1>', item)
                      if key == 'preprocessing' and isinstance(item, str)
                      else canonical_protocol(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_protocol(item) for item in value]
    return value

def raw_digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()

def digest_json(value):
    return raw_digest_json(canonical_protocol(value))

def resolve_run_config(directory, current):
    path = directory / 'config.json'
    if not path.exists():
        return current
    previous = json.loads(path.read_text())
    # These fields were added only after the original protocol hash was computed.
    metadata = {'protocol_hash', 'created_utc', 'gpu'}
    old_payload = {key: value for key, value in previous.items() if key not in metadata}
    new_payload = {key: value for key, value in current.items() if key not in metadata}
    if previous.get('protocol_hash') not in {raw_digest_json(old_payload), digest_json(old_payload)}:
        raise ValueError(f'Saved configuration hash is invalid: {path}')
    # JSON roundtrip also normalizes the integer class-label keys read back as strings.
    old = json.loads(json.dumps(canonical_protocol(old_payload), default=json_default))
    new = json.loads(json.dumps(canonical_protocol(new_payload), default=json_default))
    changes = [key for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)]
    if changes:
        details = '; '.join(f'{key}: saved={old.get(key)!r}, current={new.get(key)!r}' for key in changes)
        raise ValueError(f'Protocol changed for {directory}. Changed fields: {details}. '
                         'Restore those settings to resume, or choose a new STUDY_NAME for a different experiment.')
    # Preserve the original hash: all saved checkpoints/markers still reference it.
    # Never rewrite old checkpoints or disable verification to make resume succeed.
    if previous['protocol_hash'] != current['protocol_hash']:
        print(f'Reusing saved run identity (only preprocessing repr address changed): {directory.name}', flush=True)
    return {**current, 'protocol_hash': previous['protocol_hash']}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def atomic_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@contextlib.contextmanager
def run_lock(directory):
    path = directory / 'RUNNING.lock'
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError:
        raise RuntimeError(f'Run is locked: {path}. Confirm its previous job has stopped before removing the lock.')
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump({'pid': os.getpid(), 'host': platform.node(), 'started_utc': utc_now(),
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'user': os.environ.get('USER')}, file)
        yield
    finally:
        path.unlink(missing_ok=True)

def validate_settings():
    for name, values in [('HOLDOUTS', HOLDOUTS), ('K_VALUES', K_VALUES),
                         ('REPEATS', REPEATS), ('SEEDS', SEEDS), ('TRAINING_MODES', TRAINING_MODES),
                         ('EVAL_CONDITIONS', EVAL_CONDITIONS)]:
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{name} must be nonempty and contain no duplicates.')
    if not set(K_VALUES) <= {1, 2, 4, 7} or not set(REPEATS) <= {0, 1, 2}:
        raise ValueError('Expected k in 1,2,4,7 and dataset repeat in 0,1,2.')
    if not set(TRAINING_MODES) <= {'baseline', 'augmented'}:
        raise ValueError('TRAINING_MODES must use baseline/augmented.')
    if not set(EVAL_CONDITIONS) <= {'clean', 'blur', 'jpeg', 'resize', 'noise', 'lowpass'}:
        raise ValueError('Unknown evaluation condition.')
    if not (0 < THRESHOLD < 1) or not (1 <= CALIBRATION_BINS <= 100):
        raise ValueError('Invalid threshold or calibration bin count.')
    if any(v <= 0 for v in [IMAGE_BATCH_SIZE, HEAD_BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE]):
        raise ValueError('Training sizes, patience and learning rate must be positive.')
    if NUM_WORKERS != 0:
        raise ValueError('Use NUM_WORKERS=0 for reproducible per-epoch augmented views in this notebook.')
    if MAX_RUNS_THIS_SESSION is not None and MAX_RUNS_THIS_SESSION < 1:
        raise ValueError('MAX_RUNS_THIS_SESSION must be positive or None.')
    for key, value in AUGMENTATION.items():
        if key.endswith('_p') and not 0 <= value <= 1:
            raise ValueError(f'{key} must be in [0,1].')
        if isinstance(value, list) and (len(value) != 2 or value[0] > value[1]):
            raise ValueError(f'Invalid range: {key}')
    for config in [AUGMENTATION, EVAL_DEGRADATIONS]:
        for key, value in config.items():
            vals = value if isinstance(value, list) else [value]
            if key.endswith('_p'):
                continue
            if any(v < 0 for v in vals):
                raise ValueError(f'{key} cannot be negative.')
            if ('scale' in key or 'cutoff' in key) and any(not 0 < v <= 1 for v in vals):
                raise ValueError(f'{key} must be in (0,1].')
            if 'quality' in key and any(not 1 <= v <= 100 for v in vals):
                raise ValueError('JPEG quality must be in [1,100].')
    if not DATA_ROOT.is_dir() or not MANIFEST_ROOT.is_dir():
        raise FileNotFoundError(f'Dataset/manifests not found under {DATA_ROOT}.')
    data_resolved = DATA_ROOT.resolve()
    for path in [OUTPUT_ROOT, FEATURE_CACHE_ROOT]:
        if path.resolve().is_relative_to(data_resolved):
            raise ValueError('Outputs/caches must be outside the read-only dataset.')
        path.mkdir(parents=True, exist_ok=True)

def package_versions():
    versions = {'python': platform.python_version(), 'torch': str(torch.__version__)}
    for package in ['torchvision', 'numpy', 'pandas', 'scikit-learn', 'pillow', 'clip']:
        versions[package] = importlib.metadata.version(package)
    versions['clip_source'] = json.loads(
        importlib.metadata.distribution('clip').read_text('direct_url.json') or '{}')
    return versions

def code_fingerprint():
    # Jupyter classes often have no discoverable source file. Fingerprint the live
    # Python implementation instead; exclude filenames, line numbers, and cell IDs
    # so rerunning identical definitions gives the same protocol identity.
    import types

    def constant(value):
        if isinstance(value, types.CodeType):
            return code_record(value)
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        if isinstance(value, tuple):
            return {'tuple': [constant(v) for v in value]}
        if isinstance(value, frozenset):
            return {'frozenset': sorted([constant(v) for v in value], key=lambda v: json.dumps(v, sort_keys=True))}
        if value is Ellipsis:
            return {'ellipsis': True}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'Unsupported code constant: {type(value).__name__}')

    def code_record(code):
        record = {'bytecode': code.co_code.hex(), 'constants': constant(code.co_consts),
                  'names': code.co_names, 'variables': code.co_varnames,
                  'freevars': code.co_freevars, 'cellvars': code.co_cellvars,
                  'argcount': code.co_argcount, 'posonlyargcount': code.co_posonlyargcount,
                  'kwonlyargcount': code.co_kwonlyargcount, 'flags': code.co_flags}
        if hasattr(code, 'co_exceptiontable'):
            record['exceptions'] = code.co_exceptiontable.hex()
        return record

    def function_record(function):
        # Include decorators as well as the wrapped function, e.g. torch.no_grad.
        records, seen = [], set()
        while True:
            if id(function) in seen:
                raise ValueError('Cycle in function wrappers.')
            seen.add(id(function))
            records.append({'code': code_record(function.__code__),
                            'defaults': constant(function.__defaults__),
                            'kwdefaults': {k: constant(v) for k, v in (function.__kwdefaults__ or {}).items()}})
            if not hasattr(function, '__wrapped__'):
                return records
            function = function.__wrapped__

    def implementation_record(obj):
        if inspect.isfunction(obj):
            return {'function': function_record(obj)}
        if inspect.isclass(obj):
            methods, attributes = {}, {}
            for name, member in vars(obj).items():
                if isinstance(member, (staticmethod, classmethod)):
                    methods[name] = {'descriptor': type(member).__name__, 'function': function_record(member.__func__)}
                elif inspect.isfunction(member):
                    methods[name] = {'function': function_record(member)}
                elif name not in {'__dict__', '__weakref__', '__module__', '__doc__', '__qualname__',
                                  '__firstlineno__', '__static_attributes__'}:
                    attributes[name] = constant(member)
            return {'class': {'bases': [f'{base.__module__}.{base.__qualname__}' for base in obj.__bases__],
                              'methods': methods, 'attributes': attributes}}
        raise TypeError(f'Unsupported implementation object: {type(obj).__name__}')

    names = ['seed_everything', 'resolve_image_path', 'load_experiment', 'file_digest', 'lowpass_image', 'corrupt_image',
             'ImageTransform', 'ManifestImages', 'feature_identity', 'deterministic_features',
             'encode_batch', 'image_loader', 'export_predictions', 'evaluation_identity',
             'reliability_bins', 'classification_metrics', 'train_one', 'predict_features',
             'evaluate_one', 'epoch_features', 'train_head_epoch', 'restore_training',
             'finish_training', 'better_checkpoint']
    return digest_json({'version': ENGINE_VERSION,
                        'functions': {name: implementation_record(globals()[name]) for name in names}})
