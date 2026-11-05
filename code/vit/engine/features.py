def lowpass_image(image, cutoff):
    array = np.asarray(image, dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    mask = (np.sqrt(fx * fx + fy * fy) <= 0.5 * cutoff)[..., None]
    filtered = np.fft.ifft2(np.fft.fft2(array, axes=(0, 1)) * mask, axes=(0, 1)).real
    return Image.fromarray(np.round(np.clip(filtered, 0, 1) * 255).astype(np.uint8))

def corrupt_image(image, operation, value, rng):
    if operation == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if operation == 'jpeg':
        with io.BytesIO() as buffer:
            image.save(buffer, format='JPEG', quality=int(value), subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert('RGB').copy()
    if operation == 'resize':
        width, height = image.size
        return image.resize((max(1, round(width * value)), max(1, round(height * value))),
                            Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
    if operation == 'noise':
        array = np.asarray(image, dtype=np.float32) / 255.0
        noisy = np.clip(array + rng.normal(0, value, array.shape), 0, 1)
        return Image.fromarray(np.round(noisy * 255).astype(np.uint8))
    if operation == 'lowpass':
        return lowpass_image(image, value)
    raise ValueError(operation)

class ImageTransform:
    def __init__(self, preprocess, augmented=False, condition='clean'):
        # Official OpenAI CLIP transform: Resize, CenterCrop, RGB, ToTensor, Normalize.
        parts = preprocess.transforms
        self.geometry = transforms.Compose(parts[:2])
        self.finish = transforms.Compose(parts[2:])
        self.augmented, self.condition = augmented, condition

    def __call__(self, image, image_hash):
        image = self.geometry(image)
        if self.augmented:
            specs = [('blur', 'blur_radius'), ('jpeg', 'jpeg_quality'), ('resize', 'resize_scale'),
                     ('noise', 'noise_sigma'), ('lowpass', 'lowpass_cutoff')]
            for operation, key in specs:
                if random.random() < AUGMENTATION[operation + '_p']:
                    lo, hi = AUGMENTATION[key]
                    value = random.randint(int(lo), int(hi)) if operation == 'jpeg' else random.uniform(lo, hi)
                    image = corrupt_image(image, operation, value, np.random)
        elif self.condition != 'clean':
            # Content-based seed makes degradation independent of batch order/model/run.
            seed = int(hashlib.sha256(f'{EVALUATION_SEED}:{image_hash}:{self.condition}'.encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            keys = {'blur': 'blur_radius', 'jpeg': 'jpeg_quality', 'resize': 'resize_scale',
                    'noise': 'noise_sigma', 'lowpass': 'lowpass_cutoff'}
            image = corrupt_image(image, self.condition, EVAL_DEGRADATIONS[keys[self.condition]], rng)
        return self.finish(image)

class ManifestImages(Dataset):
    def __init__(self, frame, transform):
        self.frame, self.transform = frame.reset_index(drop=True), transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.resolved_path) as image:
            image = self.transform(image.convert('RGB'), row.sha256)
        return image, float(row.label)

ENCODER = PREPROCESS = None
ENVIRONMENT = None
ENGINE_HASH = None

def ensure_encoder():
    global ENCODER, PREPROCESS
    if ENCODER is None:
        ENCODER, PREPROCESS = clip.load(MODEL_NAME, device=DEVICE, jit=False,
                                        download_root=str(CLIP_DOWNLOAD_ROOT))
        ENCODER.eval()
        for parameter in ENCODER.parameters():
            parameter.requires_grad_(False)
    return ENCODER, PREPROCESS

def feature_identity(frame, condition):
    return {'images': frame.sha256.tolist(), 'model': MODEL_NAME,
            'preprocessing': repr(PREPROCESS), 'l2_normalized': True,
            'condition': condition,
            'degradation': EVAL_DEGRADATIONS if condition != 'clean' else None,
            'evaluation_seed': EVALUATION_SEED if condition != 'clean' else None,
            'engine': ENGINE_HASH, 'environment': ENVIRONMENT,
            'device_type': DEVICE.type, 'image_batch_size': IMAGE_BATCH_SIZE}

@torch.no_grad()
def encode_batch(images):
    features = F.normalize(ENCODER.encode_image(images.to(DEVICE)).float(), dim=-1)
    if not torch.isfinite(features).all():
        raise RuntimeError('Nonfinite CLIP features.')
    return features

def image_loader(frame, transform, shuffle=False, seed=0):
    return DataLoader(ManifestImages(frame, transform), batch_size=IMAGE_BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',
                      generator=torch.Generator().manual_seed(seed))

@torch.no_grad()
def deterministic_features(frame, condition='clean'):
    ensure_encoder()
    key = digest_json(feature_identity(frame, condition))
    path = FEATURE_CACHE_ROOT / f'{key}.pt'
    if USE_FEATURE_CACHE and path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        expected_labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32)
        if (cached['key'] != key or cached['features'].shape[0] != len(frame)
                or not torch.equal(cached['labels'], expected_labels)
                or not torch.isfinite(cached['features']).all()):
            raise ValueError(f'Invalid feature cache: {path}')
        return cached['features'], cached['labels']
    loader = image_loader(frame, ImageTransform(PREPROCESS, condition=condition))
    xs, ys = [], []
    for index, (images, labels) in enumerate(tqdm(loader, desc=f'CLIP {condition}', leave=False), 1):
        xs.append(encode_batch(images).cpu())
        ys.append(labels.float())
        report_progress(OUTPUT_ROOT, 'running', phase=f'CLIP {condition}', batch=index, batches=len(loader))
    features, labels = torch.cat(xs), torch.cat(ys)
    if USE_FEATURE_CACHE:
        atomic_torch(path, {'key': key, 'features': features, 'labels': labels})
    return features, labels
