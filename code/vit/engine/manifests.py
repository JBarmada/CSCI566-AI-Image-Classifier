VERIFIED_CONTENT = {}

def resolve_image_path(raw_path):
    text = str(raw_path).replace('\\', '/')
    marker = re.search(r'(?:^|/)GenImage_Compact(?:_shared)?/', text)
    if marker:
        path = DATA_ROOT / text[marker.end():]
    else:
        path = Path(text)
        path = path if path.is_absolute() else DATA_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f'Manifest path is outside DATA_ROOT: {raw_path}')
    return path

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_experiment(experiment):
    match = re.fullmatch(r'holdout_(.+)_k(1|2|4|7)_rep([012])', experiment)
    if not match:
        raise ValueError(f'Unexpected experiment name: {experiment}')
    holdout, k, rep = match.group(1), int(match.group(2)), int(match.group(3))
    frames, checksums = {}, {}
    for split in ['train', 'val', 'test']:
        csv = MANIFEST_ROOT / experiment / f'{split}.csv'
        checksums[split] = file_digest(csv)
        frame = pd.read_csv(csv, dtype={'sha256': str})
        required = {'path', 'label', 'source_generator', 'sha256'}
        if not required <= set(frame):
            raise ValueError(f'{csv}: missing {required - set(frame)}')
        if frame[list(required)].isna().any().any() or not frame.label.isin([0, 1]).all():
            raise ValueError(f'{csv}: missing values or invalid labels.')
        frame['label'] = frame.label.astype(int)
        frame['sha256'] = frame.sha256.str.lower()
        if not frame.sha256.str.fullmatch(r'[0-9a-f]{64}').all():
            raise ValueError(f'{csv}: invalid content hash.')
        if frame.label.value_counts().to_dict() != EXPECTED_COUNTS[split]:
            raise ValueError(f'{csv}: unexpected counts {frame.label.value_counts().to_dict()}')
        frame['resolved_path'] = frame.path.map(lambda p: str(resolve_image_path(p)))
        for row in frame.itertuples():
            path = Path(row.resolved_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if VERIFY_IMAGE_HASHES:
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns)
                if key not in VERIFIED_CONTENT:
                    VERIFIED_CONTENT[key] = file_digest(path)
                if VERIFIED_CONTENT[key] != row.sha256:
                    raise ValueError(f'Content differs from manifest: {path}')
        frames[split] = frame
    for column in ['resolved_path', 'sha256']:
        if not pd.concat([f[column] for f in frames.values()]).is_unique:
            raise ValueError(f'{experiment}: duplicate {column} within or across splits.')
    generators = {s: set(f.loc[f.label == 1, 'source_generator']) for s, f in frames.items()}
    if len(generators['train']) != k or generators['train'] != generators['val']:
        raise ValueError(f'{experiment}: training/validation generators do not match k.')
    if generators['test'] != {holdout} or holdout in generators['train']:
        raise ValueError(f'{experiment}: held-out generator leaked into training/validation.')
    nature_gens = set(pd.concat([f.loc[f.label == 0, 'source_generator'] for f in frames.values()]))
    if not nature_gens <= set(GENERATOR_FAMILIES):
        raise ValueError(f'{experiment}: unknown real-image source: {nature_gens}')
    return frames, {'experiment': experiment, 'holdout': holdout, 'k': k, 'rep': rep,
                    'train_generators': sorted(generators['train']),
                    'nature_generator': 'shared', 'real_sources': sorted(nature_gens),
                    'dataset_fingerprint': DATASET_FINGERPRINT, 'smoke_test': SMOKE_TEST,
                    'manifest_sha256': checksums}
