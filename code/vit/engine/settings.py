from pathlib import Path

# Storage: dataset is read-only; outputs and caches are written under project storage.
SHARED_ROOT = Path('/project2/yzhao010_1531/csci566-genimage')
DATA_ROOT = SHARED_ROOT / 'data/extracted/GenImage_Compact'
MANIFEST_ROOT = DATA_ROOT / 'benchmark/manifests'
STUDY_NAME = 'vit_benchmark/notebook'
OUTPUT_ROOT = SHARED_ROOT / 'results' / STUDY_NAME
FEATURE_CACHE_ROOT = SHARED_ROOT / 'features' / STUDY_NAME

# Plan: 8 holdouts x 4 k values x 3 repeats x 1 seed x 2 modes = 192 runs.
HOLDOUTS = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
            'stable_diffusion_v_1_5', 'VQDM', 'wukong']
K_VALUES = [1, 2, 4, 7]
REPEATS = [0, 1, 2]
SEEDS = [17]
TRAINING_MODES = ['baseline', 'augmented']
# Optionally restrict by exact folder name; [] means all combinations above.
EXPERIMENT_ALLOWLIST = []

# Model and classifier: same baseline defaults as the existing ViT starter.
MODEL_NAME = 'ViT-L/14'
CLIP_DOWNLOAD_ROOT = Path.home() / '.cache/clip'
IMAGE_BATCH_SIZE = 16       # Use 8 or 4 if GPU memory runs out.
HEAD_BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
NUM_WORKERS = 0             # Notebook-safe default.
REQUIRE_GPU = True

# Runtime and restart behavior. Change STUDY_NAME to intentionally start a new study.
RUN_TRAINING = True
RUN_FINAL_TEST = True       # Set False to train/validate without final test reporting.
RESUME = True
CONTINUE_ON_ERROR = False   # Fail immediately by default; errors are logged either way.
VERIFY_IMAGE_HASHES = True  # Verify manifest SHA-256 against actual bytes once per invocation.
USE_FEATURE_CACHE = True    # Deterministic features only; stochastic train views are never cached.
MAX_RUNS_THIS_SESSION = None  # e.g. 2; None runs the whole selected plan.

# Baseline has only CLIP's normal resize/center-crop/normalization.
# Augmented mode: on each training image, each operation is independently applied
# with its own probability, in the order below. Fresh views are drawn every epoch.
# Probabilities apply to BOTH real and fake images. Validation/test stay deterministic.
AUGMENTATION = {
    'blur_p': 0.3, 'blur_radius': [0.1, 2.0],
    'jpeg_p': 0.3, 'jpeg_quality': [40, 95],
    'resize_p': 0.3, 'resize_scale': [0.5, 1.0],
    'noise_p': 0.3, 'noise_sigma': [0.005, 0.03],  # pixels scaled to [0, 1]
    'lowpass_p': 0.3, 'lowpass_cutoff': [0.35, 0.9],  # fraction of axis Nyquist frequency
}

# Optional robustness evaluation adds evaluations, not training runs.
# Keep ['clean'] for the initial 24-run benchmark. Add named conditions later to
# evaluate existing checkpoints without retraining. Conditions are applied to both labels.
EVAL_CONDITIONS = ['clean']  # available: clean, blur, jpeg, resize, noise, lowpass
EVAL_DEGRADATIONS = {
    'blur_radius': 1.0, 'jpeg_quality': 60, 'resize_scale': 0.5,
    'noise_sigma': 0.02, 'lowpass_cutoff': 0.6,
}
EVALUATION_SEED = 566
CALIBRATION_BINS = 10       # Reliability diagram: mean prob_fake vs observed fake fraction.
AUROC_TOLERANCE = 1e-12
SMOKE_TEST = False
STOP_AFTER_EPOCH = None
# Report raw probability calibration: ECE, Brier, log loss. No fitted calibration.

# Expected counts per class in the dataset team's experiment CSVs.
EXPECTED_COUNTS = {'train': {0: 1400, 1: 1400},
                   'val': {0: 280, 1: 280}, 'test': {0: 280, 1: 280}}
GENERATOR_FAMILIES = {
    'ADM': 'diffusion', 'BigGAN': 'GAN', 'glide': 'diffusion',
    'stable_diffusion_v_1_4': 'diffusion', 'stable_diffusion_v_1_5': 'diffusion',
    'VQDM': 'diffusion', 'wukong': 'diffusion', 'Midjourney': 'unassigned',
}
# Midjourney's exact model/version is not specified in these manifests; do not
# silently assign it a family. Family summaries need multiple corresponding holdouts.
