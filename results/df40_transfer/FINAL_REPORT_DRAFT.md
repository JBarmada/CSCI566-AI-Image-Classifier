# Generator diversity and transfer in AI-image detection

**Team review draft — updated 26 November 2026.** This draft summarizes verified
artifacts and identifies preliminary findings. Confirm the final course format,
authorship, citations, figures and any teammate contributions before submission.

**Integration note:** The separate Claude/GPT GenImage results and extended
ViT confidence/error analysis are now available on teammate branches. They are
not incorporated as accepted findings in this draft yet. Follow the
[consolidation plan](../../docs/results-consolidation.md) to verify common
image membership, response coverage and provenance before adding their tables.

## Abstract

We study whether training on a greater number of image generators improves
detection of an unseen generator. Using corrected GenImage splits, we compare
a fully fine-tuned ImageNet-pretrained ResNet-50 and a frozen CLIP ViT-L/14
encoder with a trained linear head. Both pipelines use the same 192-run
experimental matrix. Mean clean accuracy increases with training-generator
count for both models. At k=7, the baseline CNN and ViT achieve 83.30% and
88.30% mean accuracy over eight held-out generators, respectively. The full
k=7 ViT robustness study shows that augmentation improves accuracy under JPEG
compression and resizing, although it reduces clean accuracy. For JPEG, the
accuracy gain at threshold 0.5 accompanies fewer real-image false alarms and
fewer detected fakes; mean AUROC remains nearly unchanged. The CNN
robustness comparison passed the original clean-score replay check after four
affected holdouts were rerun on V100. In a small DF40 face-manipulation
evaluation,
the original detectors have high false-positive rates. A supervised
face-adapted linear head improves results on the selected FaceForensics++
sample, but the test had been examined previously and person-level separation
is unverified. Small zero-shot vision-language and generator-objective
comparisons provide additional exploratory baselines. On the same exploratory
189-image face set, a fixed-prompt OpenAI model achieved 33.86% accuracy,
50.37% balanced accuracy and 0.5783 AUROC, accepting most real crops while
missing most manipulated crops.

## Research question and context

Our main question is whether increasing fake-generator diversity at a fixed
training-image budget improves detection on an unseen generator. Prior work
motivates both the diversity hypothesis and the risk of dataset artifacts in
AI-image detection [1–3]. We additionally test robustness to post-processing
and transfer from generated images to manipulated faces [4].

## Data and experimental design

The corrected GenImage benchmark has eight held-out generators, four values
of k (1, 2, 4, 7), three dataset repeats and baseline/augmented modes. It
contains 96 split manifests and 192 runs for each detector. Each run has
1,400 real and 1,400 fake training images, 280 of each class for validation,
and 280 of each class for test. Fake test images come from the held-out
generator. The two pipelines share the image manifests. The same test images
and some real partitions recur across settings, and all runs use seed 17.
Dataset repeats are not independent classifier-seed trials.

The CNN is an ImageNet-pretrained ResNet-50 fine-tuned for binary detection.
The ViT uses a frozen CLIP ViT-L/14 image encoder and a supervised linear
classifier [5]. Augmented training uses blur, JPEG, resize, noise and
low-pass operations. These are complete-pipeline comparisons, so pretraining,
architecture and fine-tuning differ. Both pipelines use threshold 0.5 for
the standard benchmark. For each k and mode, we average the three repeats
within each holdout and then weight the eight holdouts equally.

## Clean held-out-generator results

| k | CNN baseline accuracy | CNN augmented accuracy | ViT baseline accuracy | ViT augmented accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62.75% | 58.39% | 73.00% | 70.74% |
| 2 | 70.25% | 64.92% | 77.29% | 73.07% |
| 4 | 80.86% | 69.90% | 85.42% | 80.43% |
| 7 | 83.30% | 72.24% | 88.30% | 82.78% |

All 192 CNN and 192 ViT runs passed their original audits. An additional
matched analysis checked 107,520 test-image pairs by SHA-256 and label and
recomputed accuracy, AUROC, Brier score and 10-bin expected calibration error.
At k=7 baseline, mean AUROC was 0.9007 for CNN and 0.9564 for ViT. Mean
10-bin ECE was 0.1138 for CNN and 0.2816 for ViT; calibration error depends on
the binning and aggregation method. These summaries do not establish an
architecture-only effect.

## Compression and resize robustness

The k=7 evaluation applies JPEG quality 60 or a bicubic 224→112→224 resize
after the common geometry and before normalization. The completed ViT study
contains 48 clean references and 96 degraded evaluations. ViT baseline mean
accuracy was 88.30% clean, 69.24% JPEG and 56.96% resize. ViT augmented mean
accuracy was 82.78%, 75.99% and 81.85%, respectively.

The JPEG accuracy gain conceals an operating-point tradeoff. At threshold 0.5,
equal-weight mean real-image false-positive rate fell from 54.11% to 17.02%,
while fake recall fell from 92.59% to 69.00%. Mean AUROC changed from 0.84987
to 0.84941. These comparisons use separately trained models, and the same test
images recur across repeats. A descriptive test-ROC comparison at 10% empirical
false-positive rate gives 58.78% baseline versus 60.49% augmented mean recall;
it uses the test labels to find operating points and is not independent
threshold calibration. See the [paired analysis](poster_assets/JPEG_ERROR_ANALYSIS.md).

The matching CNN study covers all eight holdouts. After an initial P100 replay
failed the 0.001 maximum-score tolerance for four holdouts, the affected 24
checkpoints were rerun on V100. All reproduced their original clean scores
exactly, and the 72 new clean/JPEG/resize evaluations passed an independent
audit. The original P100 results are preserved. The validated CNN baseline
mean accuracy was 83.30% clean, 64.43% JPEG, and 50.20% resize; the augmented
CNN achieved 72.24%, 70.65%, and 68.85%, respectively. These are equally
weighted averages across eight holdouts after averaging three repeats. The CNN
shows the same JPEG operating-point tradeoff as the ViT: at threshold 0.5,
real false positives fell from 65.79% to 20.01% while fake recall fell from
94.64% to 61.32%. Both class-specific rates fell in all eight holdout means
for each pipeline. These descriptive results compare separately trained
models and do not isolate architecture or establish a cause. See the
[class-specific analysis](poster_assets/JPEG_ERROR_ANALYSIS.md). GPU
differences explain the observed replay mismatch in the tested setup; see
[the audit](cnn_robustness_v100/README.md).

## Face-manipulation transfer

In a subsequent exploratory 189-image FaceForensics++-derived comparison,
the five frozen model variants were tested on 54 real crops and 27 crops for
each of five manipulation labels. A P100 replay passed reference checks. At
threshold 0.5, the baseline CNN, baseline ViT and face-adapted ViT head
flagged 53, 42 and 20 of the same 54 real crops, respectively. Fake-detection
counts varied by method; all counts and the preselected validation-threshold
results are in the [five-method results](new_data/fresh_candidate_results_v3/README.md).
Person-level independence, source roles and some mirror method labels remain
unverified; three video-ID pairs were excluded before scoring. The adapted
head trained on three evaluated methods, so this small follow-up does not
establish a reliable general adaptation improvement.

A subsequent zero-shot OpenAI Responses API evaluation used model
`gpt-6.1-terra` on those exact 189 image hashes, one image per request, with a
fixed face-manipulation prompt and threshold 50/100. All predictions and the
1,134-row, six-model comparison passed the independent result audits; local
regeneration reproduced every summary/comparison CSV exactly. No retraining,
prompt tuning or threshold tuning was performed for this run.

| Model/configuration | Accuracy | Balanced accuracy | AUROC | Real falsely flagged /54 | Fakes detected /135 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI, zero-shot | 33.86% | 50.37% | 0.5783 | 6 | 16 |
| CNN, baseline | 70.90% | 50.19% | 0.5690 | 53 | 133 |
| CNN, augmented | 60.85% | 55.37% | 0.5227 | 31 | 92 |
| ViT, baseline | 62.43% | 50.37% | 0.4930 | 42 | 106 |
| ViT, augmented | 58.73% | 49.44% | 0.5221 | 39 | 96 |
| ViT, face-adapted (supervised reference) | 69.84% | 67.78% | 0.7583 | 20 | 98 |

OpenAI's real-image false-positive rate was 11.11%, but its fake recall was
only 11.85%: it missed 119/135 manipulated crops. It detected 6/27 SimSwap,
3/27 FOMM, 3/27 Wav2Lip, 3/27 InSwapper and 1/27 PIRender crops. The baseline
CNN and ViT showed more real false alarms and higher fake recall. Because
135/189 images are fake, always predicting fake would achieve 71.43% accuracy
but 50% balanced accuracy; raw accuracy does not establish a useful detector
or a model ranking. The 54 real crops are shared across all method summaries,
and the crops are linked by source-video groups. OpenAI scores are prompted
estimates, not calibrated probabilities. This single model/prompt comparison
does not resolve the candidate's identity or method-label limitations. The
[complete exports and analysis](openai_face/df40_five_method_v3_v2/README.md)
preserve the protocol, raw scored responses, per-method metrics and audits.

An additional exploratory still-photo test used 50 IMDB-WIKI real photos and
50 matched-path outputs for each of InsightFace and Stable Diffusion inpainting
from DeepFakeFace. At threshold 0.5, the baseline CNN falsely flagged 23/50
reals and detected 31/50 InsightFace and 34/50 inpainting samples; the baseline
ViT falsely flagged 4/50 and detected 28/50 and 33/50, respectively. The
augmented models and AUROC values are in the [still-photo results](still_face_v1/README.md).
The differing samples cannot explain the DF40 real-image errors. In this sample
all fake files were 512x512 while real files varied in size, introducing a
potential class-correlated shortcut. Source-image and identity overlap and
successful face edits are not independently established.
In a paired follow-up applying a common square crop, 512x512 resize, and JPEG
encoding to all images, baseline CNN real false positives changed from 23/50
to 28/50 and ViT from 4/50 to 6/50. This combined intervention cannot isolate
the effects of any one processing operation or identify camera effects; see
[the processing follow-up](still_face_uniform_v1/README.md).
Subsequent separate, post-hoc geometry-only and JPEG-quality-95 tests showed
that geometry changed five baseline CNN and one baseline ViT real decisions,
while that JPEG variant changed none. This is limited to these 50 reals and
these settings; see [factor analysis](still_face_geometry_v1/README.md).

The first DF40 subset contains 100 real Celeb-DF crops, 200 SimSwap, 100 FOMM
and 100 Wav2Lip crops. The 100 real images are reused for each per-method
comparison. Existing CNN and ViT models were evaluated without retraining.
At the original threshold, real-image false-positive rates were 99% for
baseline CNN, 57% for augmented CNN, 90% for baseline ViT and 83% for
augmented ViT. The fixed-prompt zero-shot CLIP baseline produced 24% false
positives, but detected 43% of SimSwap, 60% of FOMM and 27% of Wav2Lip.
Results apply to this subset and do not represent all DF40 methods.

A second subset uses the FaceForensics++ portion of DF40. Validation and test
each contain 70 real images and 70 fake images per method. The selection keeps
known video-ID pairs in the same partition; person-level independence is not
established. Thresholds targeting at most 10% validation-real false positives
were frozen before test inference. A new supervised linear head trained on
separate FF crops achieved 4/70 test real false positives and detected 31/70
SimSwap, 51/70 FOMM and 15/70 Wav2Lip images. The earlier zero-shot CLIP
baseline had 9/70 false positives and detected 21/70, 24/70 and 11/70,
respectively. All three fake methods appeared in face-adaptation training.
The FF test was previously inspected, so this is an exploratory follow-up.
The adapted-head job's final summary write failed; a separate replay reproduced
its validation selection and test predictions exactly, and the failure is
recorded in the experiment log.

## Exploratory model comparisons

An open Qwen2.5-VL-3B vision-language model was evaluated using one fixed
prompt on 32 real and 32 ADM fake images from the shared test set. It answered
all 64 and achieved 42/64 accuracy, with 1/32 real false positives and 11/32
fake detections. The matched baseline CNN and ViT achieved 48/64 and 61/64,
respectively. This one-model, one-prompt, one-generator sample is too small
for broad rankings. Image preprocessing differs among model families.

On the k=7 repeat-0 frozen CLIP features for all eight holdouts, a common
feature adapter was trained with binary loss, a generator-adversarial loss,
or supervised contrastive loss. After the first seed, two optimization seeds
were added. Averaging three seeds within each holdout and then across holdouts,
test accuracy was 0.9162, 0.9185 and 0.9150, with mean AUROC 0.9742, 0.9741
and 0.9742, respectively. Adversarial accuracy was higher in five holdouts and
lower in three. The extension and additional seeds were decided after seeing
previous results; the test sets were previously examined, and just one dataset
repeat was used per holdout. These exploratory findings do not establish a
reliable improvement. The CLIP backbone was frozen.

## Limitations and next checks

The test data are reused across several analyses and selection decisions.
Celebrity identities and source videos may overlap within DF40 subsets.
The held-out FF video IDs are separated, but person independence is unknown.
Fake labels follow released dataset provenance and do not prove that each
frame exhibits a successful manipulation. Face crop preprocessing and source
differences can create confounding cues. Qwen and OpenAI each represent one
model with one prompt on small selected subsets, not a broad ranking of open
or proprietary vision-language systems. OpenAI's low false-positive rate
accompanied low fake recall; a passed output audit does not certify the
underlying dataset labels or independence. The CNN
robustness rerun passed the original replay tolerance; the earlier P100 results
and their discrepancy are retained for provenance. The regularized objectives
need more dataset repeats and a fresh test before claims about improvement.

## References

1. Zhu et al. [GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image](https://papers.neurips.cc/paper_files/paper/2023/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf). NeurIPS 2023.
2. Park and Owens. [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html). CVPR 2025.
3. Grommelt et al. [Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/abs/2403.17608). 2024.
4. Yan et al. [DF40: Toward Next-Generation Deepfake Detection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/34239f60eca7ce9bee5280aaf81362d8-Abstract-Datasets_and_Benchmarks_Track.html). NeurIPS 2024.
5. Radford et al. [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html). ICML 2021.

Detailed provenance, raw scores, model-selection records and the current error
status are linked from [the experiment log](FINAL_REPORT_EXPERIMENT_LOG.md).
