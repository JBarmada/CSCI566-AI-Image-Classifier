# CSCI566 - AI-Generated Image Detection

Does training on more image generators improve detection of an unseen generator?
This project compares a fully fine-tuned, ImageNet-pretrained ResNet-50 with a
frozen CLIP ViT-L/14 encoder and supervised linear classifier. Both models use
the same corrected GenImage manifests, held-out-generator tests and experiment
matrix. The number of fake-image generators represented during training varies
across **k = 1, 2, 4 and 7**, while the total number of training images remains
fixed.

## Current status

**Verified November 10, 2026: both corrected clean benchmarks are complete.**
The ViT and CNN studies each contain 192 runs and passed their final CARC audits.
The published ViT export independently checked all **384 validation/test
prediction files (215,040 rows)** against the exact manifests, labels and
recorded metrics. The audited CNN results and checkpoints currently remain on
CARC and still need a small, shareable analysis export for matched reporting.

| Component | Status |
|---|---|
| Corrected dataset | 96 experiments, 288 split CSVs, fixed test membership per holdout, shared real partitions, release hashes and acceptance checks |
| ViT clean benchmark | 192/192 completed and audited: 8 holdouts x 4 k values x 3 repeats x 2 training modes |
| Published ViT analysis | Per-run results/settings, image-level predictions, training histories, calibration, paired augmentation differences and figures |
| CNN clean benchmark | 192/192 completed and audited; checkpoint hashes, frozen split membership, predictions, labels and metrics verified |
| CNN k=7 robustness | Evaluation notebook ready for clean, JPEG-60 and 224→112→224 bicubic tests; default selection is ADM across repeats 0–2 |
| Remaining analysis | Run/expand CNN robustness, export audited CNN summaries, perform matched CNN–ViT analysis and finish the report/poster |

The accepted clean studies do not need retraining. Robustness evaluation reuses
saved checkpoints and must not modify the clean benchmark outputs.

## Start here

| Need | Read |
|---|---|
| Completed ViT results, data dictionary and interpretation | [Benchmark analysis handoff](results/vit_benchmark/README.md) |
| Historical November 8 status and suggested team responsibilities | [November 8 review](docs/vit-status-2026-11-08.md) |
| Dataset design, acceptance checks and reproducibility | [Corrected benchmark guide](docs/benchmark.md) |
| Exact manifests and raw analysis archives | [Drive handoff](https://drive.google.com/drive/folders/155zGH4KJr_cifkuqI29l0BbEocfMlb74) |
| CARC environment and shared storage | [CARC setup](docs/carc.md) |
| ViT notebook and runner | [ViT guide](docs/vit.md) |
| CNN workflow | [CNN notebook](cnn.ipynb) and [CARC batch guide](docs/cnn.md) |
| CNN k=7 JPEG/resize robustness evaluation | [Robustness notebook](CNN_K7_Robustness_Evaluation.ipynb) |
| Original dataset workflow | [Dataset notebook](GenImage_Final_Dataset_Workflow.ipynb) |
| Earlier ADM-only study | [Archived pilot results](results/vit_master_v1/README.md) |

**Drive access:** the handoff was uploaded to the connected account's My Drive.
At verification it was private; creation under the original shared dataset
folder returned `403 insufficientParentPermissions`. An editor must share or
move the handoff so teammates can access it.

## Experimental design

The corrected benchmark contains 96 experiment manifests: eight held-out fake
generators, four k values and three repeats. Each manifest has 1,400 real +
1,400 fake training images, 280 + 280 validation images and 280 + 280 test
images. The test fake images always come from the held-out generator. Test
membership and the real partitions are fixed for a holdout so changes across k
are not caused by changing the test set.

For k=1, 2 and 4, repeats use different combinations of training generators.
For k=7, only one generator combination exists, so repeats vary the sampled fake
training/validation images. Each model runs baseline and augmented training with
seed 17, giving **8 x 4 x 3 x 2 = 192 runs per model**. Labels are real=0 and
fake=1; reported accuracy uses a fixed 0.5 threshold and AUROC uses raw fake
probabilities.

| Pipeline | CNN | ViT |
|---|---|---|
| Backbone | ResNet-50, ImageNet-1K V2 weights | CLIP ViT-L/14 |
| Training | Fine-tune the network and replace the final layer with one binary logit | Freeze the encoder and train a supervised linear head |
| Input | Bicubic resize, 224 center crop, ImageNet normalization | CLIP preprocessing |
| Baseline | Deterministic preprocessing only | Deterministic CLIP preprocessing only |
| Augmented | Independent blur, JPEG, resize, noise and low-pass operations during training | Matching operation families during training |
| Validation | Checkpoint selection from validation performance; test data is not used for selection | Same separation of validation selection and final testing |

This is a comparison of complete pipelines, not a controlled backbone-only
ablation. The models have different pretraining and fine-tuning strategies, so
results should be interpreted with those differences stated explicitly.

## Clean ViT results

Each value is the equally weighted mean across eight held-out generator variants,
after averaging three dataset repeats within each holdout. Accuracy uses a fixed
decision threshold of 0.5.

| Training generators (k) | Baseline accuracy | Augmented accuracy | Baseline AUROC | Augmented AUROC |
|---|---:|---:|---:|---:|
| 1 | 73.00% | 70.74% | 0.8608 | 0.8420 |
| 2 | 77.29% | 73.07% | 0.9012 | 0.8790 |
| 4 | 85.42% | 80.43% | 0.9412 | 0.9212 |
| 7 | 88.30% | 82.78% | 0.9564 | 0.9361 |

Average clean performance improves with k for both modes; individual holdouts
can behave differently. The combined augmentation recipe scores lower on clean
tests at each k. Its intended robustness benefit still needs degraded-image
evaluation. Raw probability calibration also remains imperfect.

![Clean ViT performance versus training-generator count](results/vit_benchmark/figures/performance_vs_k.png)

The figures' error bars show variation across holdout means. Dataset repeats
reuse test images and real partitions, and every run uses classifier seed 17.
Account for these dependencies when estimating uncertainty. Stable Diffusion
1.4 and 1.5 are related variants. JPEG settings are matched, while compression
history, image size, semantic content and undetected related images remain
potential confounds. Conclusions concern this corrected benchmark.

## Clean CNN benchmark

The corrected ResNet-50 study completed and passed its final audit on CARC on
November 10, 2026. The audit accepted all 192 runs only after verifying:

- the completion marker, configuration and summary share one protocol hash;
- each selected `best.pt` matches its recorded SHA-256 checksum;
- archived train/validation/test inputs match the frozen manifests;
- image identities do not leak across splits;
- validation/test predictions contain the expected unique images and labels;
- saved accuracy, F1, AUROC, calibration and confusion counts recompute from the
  prediction files.

The source-of-truth CNN directory is
`results/cnn_benchmark/notebook/` on CARC. `CNN_AUDIT.json` records acceptance;
`all_experiment_results.csv`, `aggregate_by_holdout_k_and_variant.csv` and
`equal_weight_holdout_summary.csv` provide the clean-study tables. These
corrected exports are not the removed ADM-only pilot tables.

## k=7 robustness evaluation

[`CNN_K7_Robustness_Evaluation.ipynb`](CNN_K7_Robustness_Evaluation.ipynb)
evaluates existing corrected-benchmark checkpoints without retraining. It makes
separate passes over the same held-out images under:

- `clean`: standard resize/crop only;
- `jpeg60`: JPEG quality 60 with 4:2:0 subsampling;
- `resize_112`: bicubic 224 → 112 → 224.

The condition is applied after the standard 224 resize/center crop and before
tensor conversion and normalization, to both real and fake images. Results are
written under `results/cnn_robustness_k7_v1/`, separate from clean training.
The notebook defaults to ADM repeats 0–2 and both baseline/augmented variants;
edit `HOLDOUTS` deliberately when expanding the comparison to other holdouts.
Report accuracy and AUROC with holdout, repeat, variant and condition.

## Find the analysis data

| File or folder | Use |
|---|---|
| [All clean test results](results/vit_benchmark/summary/all_results.csv) | 192 rows with classification/calibration metrics, confusion counts and protocol hashes |
| [Full run details](results/vit_benchmark/summary/run_details.csv) | Test/validation metrics, training-generator combinations, best epochs, settings and environment |
| [Per-holdout summaries](results/vit_benchmark/summary/by_holdout_k.csv) | Mean, SD and count across dataset repeats |
| [Macro summaries](results/vit_benchmark/summary/macro_across_holdouts.csv) | Equal-weight holdout averages and variation |
| [Paired augmentation differences](results/vit_benchmark/summary/augmentation_deltas.csv) | Augmented-minus-baseline differences within each experiment |
| [Prediction exports](results/vit_benchmark/predictions/) | Gzip CSVs containing all validation/test probabilities and image identities |
| [Training/calibration metadata](results/vit_benchmark/metadata/) | Epoch histories, reliability bins, original run records, audit reports and checksums |
| [Figures](results/vit_benchmark/figures/) | Performance versus k, holdout AUROC heatmaps and raw reliability |

Audited CNN files currently live on CARC rather than in this GitHub checkout:

| CARC path under `results/cnn_benchmark/notebook/` | Use |
|---|---|
| `CNN_AUDIT.json` | Final acceptance status and audited run count |
| `all_experiment_results.csv` | One row per corrected CNN run |
| `aggregate_by_holdout_k_and_variant.csv` | Mean/SD/count across repeats |
| `equal_weight_holdout_summary.csv` | Equal-weight averages across holdouts |
| `runs/<experiment>__<variant>__seed17/` | Config, frozen inputs, checkpoint, history, predictions and metrics |

Read compressed tables directly with `pandas.read_csv(path)`. Keep validation and
test analyses separate. For model comparisons, verify the dataset release and
image coverage, then join by experiment, seed, split and SHA-256 `image_id`
within the selected training treatments. Check labels before computing paired
metrics. The incompatible ADM-only CNN pilot tables have been removed; collect
the corrected-benchmark CNN exports from `results/cnn_benchmark/notebook/`.
Document each model's actual preprocessing, augmentation and training settings.

## Exact dataset and CARC artifacts

The benchmark uses eight generator variants: ADM, BigGAN, GLIDE, Midjourney,
Stable Diffusion 1.4, Stable Diffusion 1.5, VQDM and Wukong. One variant is held
out from fake training and validation. Each experiment has 1,400 real + 1,400
fake training images; validation and test each contain 280 real + 280 fake images.

Large artifacts live under `/project2/yzhao010_1531/csci566-genimage`:

| Artifact | Relative path |
|---|---|
| Image root | `data/extracted/GenImage_Compact/` |
| Exact benchmark split CSVs | `data/extracted/GenImage_Compact/benchmark/manifests/` |
| Release plan, fingerprint and image inventory | `data/extracted/GenImage_Compact/benchmark/` |
| Full ViT results and saved classifier heads | `results/vit_benchmark/<holdout>/runs/` |
| ViT frozen-feature cache | `features/vit_benchmark/<holdout>/` |
| Corrected CNN outputs | `results/cnn_benchmark/notebook/` |
| CNN k=7 robustness outputs | `results/cnn_robustness_k7_v1/` |
| Earlier ViT pilot | `results/vit_master_v1/` |

The Drive manifest archive contains the exact `benchmark/` directory and its
hashes. It contains image lists; obtain the images separately and preserve their
relative paths. Real images use the original generator folders; fake images use
the existing `_release_v2/jpeg_ai/` derivatives. Verify image hashes before
training on a new machine. Images, feature caches and checkpoint binaries remain
on CARC; the GitHub handoff contains analysis exports. Preserve a durable backup
of the accepted checkpoints.

## Repository map

| Path | Purpose |
|---|---|
| `GenImage_Final_Dataset_Workflow.ipynb` | Original compact-dataset workflow |
| `code/benchmark/` | Corrected manifest construction and independent dataset audit |
| `ViT_Master_Experiments.ipynb` | Thin interactive entry point for ViT |
| `code/vit/engine/` | ViT settings, manifests, features, training, metrics and summaries |
| `code/vit/*.slurm` | ViT smoke, array training and final-audit launchers |
| `cnn.ipynb` | Single CNN training and audit implementation |
| `code/cnn/` | CNN batch runner, status tool and final-audit launchers |
| `CNN_K7_Robustness_Evaluation.ipynb` | Inference-only CNN robustness evaluation |
| `tests/` | Batch, resume, Slurm and engine regression tests |
| `archive/` | Historical notebooks retained for provenance, not current runs |

The larger number of ViT files reflects modularization of notebook logic and
separate operational roles; it does not represent extra training stages. The
core path is notebook → engine → batch runner → final audit. Smoke, status,
export and plotting scripts support validation and reporting.

## Reproduce the workflow

The [master notebook](ViT_Master_Experiments.ipynb) and
[batch runner](code/vit/run_vit_master.py) use `code/vit/engine/`.
The [benchmark guide](docs/benchmark.md) documents environment setup, preflight,
isolated GPU smoke checks, training/resume and the final audit. Coordinate study
ownership before launching work; each holdout permits one output writer.
Earlier starter notebooks are kept under [archive/vit/](archive/vit/).

The CNN benchmark uses [`cnn.ipynb`](cnn.ipynb) as its single implementation.
The Slurm array assigns one holdout to each task and limits concurrency to two:

```bash
ARRAY_JOB_ID=$(sbatch --parsable code/cnn/run_benchmark.slurm)
python code/cnn/benchmark_status.py --watch
AUDIT_JOB_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_JOB_ID" \
  code/cnn/finalize_benchmark.slurm)
```

Set `CNN_ENV_PREFIX` to an environment prefix or `CNN_PYTHON` to a verified
interpreter before submission when the defaults do not exist. Completed runs are
validated and skipped; interrupted runs resume from the latest atomically saved
epoch. The final audit is independent and must pass before results are accepted.
Do not edit or update a checkout while its jobs are active. Full commands and
recovery details are in the [CNN CARC guide](docs/cnn.md).

To recreate analysis tables from extracted handoff archives:

```bash
python code/vit/export_analysis.py --source PATH/vit_benchmark --benchmark PATH/benchmark --output results/vit_benchmark
python code/vit/plot_analysis.py --root results/vit_benchmark
```

See the [current work review](docs/vit-status-2026-11-08.md) for robustness,
matched CNN comparisons, uncertainty, calibration and reporting tasks. The
[older team plan](docs/team-plan.md) preserves the historical planning snapshot.

## Interpretation limits

- Dataset repeats reuse fixed test images and are not independent test samples.
- Stable Diffusion 1.4 and 1.5 are related variants; eight holdouts are not eight
  fully independent model families.
- Matching JPEG tables does not equal matching compression history, resolution,
  semantics or source distribution.
- The augmentation comparison is one combined recipe, not an ablation of each
  operation.
- Robustness conclusions require the same saved models, image identities,
  condition order and metrics for CNN and ViT.
- The study detects images within this corrected benchmark; it does not establish
  universal detection of arbitrary generated or edited images.
