# Independent team readiness audit — November 26, 2026

**Recommendation: start the poster and final report now.** The major experiments in the proposal and midterm are complete. Finish the evidence consolidation and claim review in parallel. This audit adds a CNN-inclusive comparison using saved outputs from the shared Drive and the published corrected ViT archives; it required no training or provider calls.

Reviewed GitHub main at `b5c0f483c463db50e3a5fd4835e1a3f4c8c7da69`. The five active frontier, confidence and matched-comparison branches were all ancestors of main. The latest commit reports 88 passing tests and 35 passing subtests. This review independently recomputed the frontier and face metrics from saved predictions; it did not rerun the training suite or historical APIs.

## Proposal and midterm coverage

| Work | Review status |
| --- | --- |
| Fixed-budget generator diversity, CNN vs ViT | Complete (192 matched run pairs, all 8 holdouts); main research result. |
| Augmentation and robustness | Complete for JPEG60 and resize at k=7; accepted V100 CNN replay corrected the historical discrepancy. |
| Confidence/calibration and errors | Complete for raw CNN/ViT metrics and extended ViT validation-fitted calibration and error analysis. |
| Deepfake transfer | Complete as an exploratory study on the frozen five-method set, including the GPT/Claude comparisons below. |
| Multimodal GenImage | Completed 90-source Gemini/Groq-Qwen pilot, 630-source Claude/GPT three-view runs (46 explicit GPT invalid scores). The new offline matched comparisons below include CNN and ViT. |
| Domain-adversarial/contrastive learning | Exploratory feature-adapter studies complete (all eight holdouts, three optimization seeds, one dataset repeat); no established winning objective. |

See the [submitted-project results draft](../../df40_transfer/FINAL_REPORT_DRAFT.md), [team contribution and merge review](../../../docs/project-review-2026-11-26.md) and [analysis guide](../../../docs/results-consolidation.md).

## New matched GenImage comparisons

The CNN archive is in the [team's verified CNN Drive folder](https://drive.google.com/drive/folders/1S12PSJPnaA62aw6icbARC7szwWYFYxrJ). All 16 k=7/repeat-0 baseline and augmented test CSVs were read and their SHA-256 checksums matched the archived checksum manifest. Each contained 560 unique test rows, 280 of each label, at threshold 0.5, seed 17. Eight locally available corrected ViT gzip archives matched the checksums in the [integration review](../openai/integration_review/openai_vs_vit_matched.csv); both modes were selected with k=7/repeat-0, clean test only.

For each holdout, joined its 70 real and 70 fake sources from the [630-image list](../claude/shared_image_list.csv) to the two CNN and two ViT configurations by SHA-256 and label. All 16 matched holdout/mode checks contained 140 sources before excluding invalid GPT outputs. All saved detector decisions matched the threshold. For each frontier comparison intersected valid scores for all participating models before computing metrics. No scores were imputed or thresholds tuned. Equal-weight mean of eight holdout metrics; real images recur in each holdout, so the row count is not the independent image count. Per-holdout counts and metrics are in [matched_by_holdout.csv](matched_by_holdout.csv).

### Larger shared sample, detector224 view

612/630 unique sources have valid GPT and Claude scores on this view (70 real, 542 fake; 1,102 holdout rows per configuration). The remaining 18 GPT scores are invalid and all are fake images. GPT and Claude receive their own 224-pixel preparation; CNN and ViT use their own existing clean preprocessing. The comparison controls source membership and approximate image geometry, not identical internal model tensors.

| Model/configuration | Fake recall | Real false-alarm rate | Balanced accuracy | Mean holdout AUROC |
| --- | ---: | ---: | ---: | ---: |
| ViT baseline | 82.67% | 3.93% | 89.37% | 0.9680 |
| CNN baseline | 80.45% | 9.29% | 85.58% | 0.9219 |
| ViT augmented | 69.71% | 2.14% | 83.79% | 0.9516 |
| Claude detector224 | 55.59% | 0.00% | 77.79% | 0.9520 |
| CNN augmented | 65.38% | 13.75% | 75.82% | 0.8444 |
| GPT detector224 | 26.80% | 0.00% | 63.40% | 0.9303 |

### Four-frontier pilot, original view

88/90 common valid source images (10 real, 78 fake; 158 holdout rows per configuration). GPT and Claude use their original view; Gemini and Groq-Qwen use the historical pilot preparation and fixed scores. Provider prompts/configurations and image processing need not be identical. The result is a descriptive small comparison, not a leaderboard over frontier models in general.

| Model/configuration | Fake recall | Real false-alarm rate | Balanced accuracy | Mean holdout AUROC |
| --- | ---: | ---: | ---: | ---: |
| ViT baseline | 86.11% | 0.00% | 93.06% | 0.9838 |
| Claude original | 77.92% | 0.00% | 88.96% | 1.0000 |
| ViT augmented | 70.28% | 0.00% | 85.14% | 0.9651 |
| Gemini | 67.64% | 0.00% | 83.82% | 0.9781 |
| CNN baseline | 80.56% | 20.00% | 80.28% | 0.9107 |
| GPT original | 52.78% | 0.00% | 76.39% | 0.9946 |
| Groq-Qwen | 62.64% | 10.00% | 76.32% | 0.8333 |
| CNN augmented | 62.36% | 20.00% | 71.18% | 0.8197 |

### Paired image-presentation comparison

For the 590 source images with valid scores across both models and all three views (70 real and 520 fake), the following metrics use the same image IDs in every row. Both models flagged 0/70 real photos in every view.

| Model | View | Fakes detected | Fake recall | AUROC |
| --- | --- | ---: | ---: | ---: |
| GPT | original | 284/520 | 54.62% | 0.9907 |
| GPT | square | 281/520 | 54.04% | 0.9875 |
| GPT | detector224 | 142/520 | 27.31% | 0.9333 |
| Claude | original | 388/520 | 74.62% | 0.9998 |
| Claude | square | 388/520 | 74.62% | 0.9962 |
| Claude | detector224 | 291/520 | 55.96% | 0.9532 |

The shape-only baseline in the [Claude view report](../claude/claude_api_view_comparison.csv) separates 629/630 original images with AUROC 0.9929: all 560 fakes are square, 69/70 real photos are non-square. Square-cropping preserves most of the observed ranking, while the 224-pixel view substantially reduces fake recall. The 224 intervention changes crop/resize and image presentation, so these results do not isolate resolution alone or prove that the models learned forgery features. GPT has 46 invalid image-view outputs (all fake, across 13 original, 15 square, 18 detector224); report coverage next to conditional metrics. The original GPT attempt-level JSONL remains unavailable in the reviewed GitHub record.

## Face comparison

Recomputed the published [GPT six-configuration table](../../df40_transfer/openai_face/df40_five_method_v3_v2/comparison/matched_predictions.csv) and [Claude six-configuration table](../../df40_transfer/claude_face/df40_five_method_v3_v1/comparison/matched_predictions.csv) against the [frozen manifest](../../df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv). All seven distinct configurations cover the exact 189 unique image hashes and labels. The five detector configurations shared between exports have identical scores/decisions.

| Model/configuration | Real false alarms | Fakes detected | Balanced accuracy | AUROC |
| --- | ---: | ---: | ---: | ---: |
| openai/zero shot | 6/54 | 16/135 | 50.37% | 0.5783 |
| cnn/baseline | 53/54 | 133/135 | 50.19% | 0.5690 |
| cnn/augmented | 31/54 | 92/135 | 55.37% | 0.5227 |
| vit/baseline | 42/54 | 106/135 | 50.37% | 0.4930 |
| vit/augmented | 39/54 | 96/135 | 49.44% | 0.5221 |
| vit/face adapted | 20/54 | 98/135 | 67.78% | 0.7583 |
| claude/zero shot | 42/54 | 121/135 | 55.93% | 0.6198 |

The dataset has 54 real crops and 135 manipulated crops (27 for each of five methods), linked to 27 recorded video pairs. At the frozen threshold, the unadapted baseline CNN is almost always fake and GPT is usually real; Claude also has many false alarms. The adapted ViT is a supervised reference and has prior test-inspection and identity limitations. Keep model-specific error counts, source/identity and mirror-label caveats; a fresh person-independent benchmark would be a future study.

## Confidence analysis review

The [new ViT analysis](../../analysis/README.md) uses all 192 runs and its [verification record](../../analysis/calibration/verification.json) confirms raw metric agreement. The recalibration code fits temperature, Platt scaling and thresholds on validation predictions only. The checked macro table shows baseline ECE at k=7 falling from 0.2816 to 0.0676 and log loss from 0.5338 to 0.2911 without changing class decisions at 0.5. At k=1, log loss worsens from 0.5861 to 1.0429. This supports a descriptive claim that calibration transfer differs across generator diversity settings.

One uncertainty caveat: the paired bootstrap reuses sampled images over the three repeats within each holdout, but draws the shared real images separately across holdouts. A confidence interval for the overall aggregate should preserve the shared-real dependency across holdouts. Use the observed descriptive differences now or revise that bootstrap before quoting its uncertainty interval. Folder-level error patterns do not establish camera effects.

## Remaining work for the team

1. **Poster and report can start.** Agree on the central story. A candidate is whether training on more generators generalizes when image presentation and task change. The core clean benchmark anchors the story; the paired 224-view result, calibration, and face errors illustrate its limits. Review actual figures together.
2. **Review and adopt the new saved-score matches.** Shriya/Aditya can integrate the CSV above into the consolidated analysis, retain coverage and separate study protocols, and confirm the original GPT request-attempt JSONL (Jeslyn). The log is needed for independent retry/settings/usage claims; if unavailable, disclose the gap and do not claim those details were audited.
3. **Final evidence and presentation checks.** Verify final captions, citations, counts, data provenance and controls; prepare the demo, report and contribution statement. Extra fresh data, expanded provider runs, CNN recalibration and new training are optional and do not block the deliverables.

The public course schedule lists the November 30 poster/demo and December 7 report. Confirm current Piazza/Gradescope submission instructions with the team.

## CNN source verification

All checksum values below match the original [SHA256SUMS.txt](https://drive.google.com/file/d/1PT_I42hQ6AUJ71d5KudPF74mqJK3tbVb/view). These are saved prediction CSVs, not new inference.

| Run | Source | SHA-256 |
| --- | --- | --- |
| holdout_ADM_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/1XglgcIVyEWL5V-a_TQmfiC2OyyIovYk_/view) | `0d1255fc1955f4f829c3f75390d22f3feb409eb65d11faaa89e5684d2f0925fe` |
| holdout_ADM_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/18PHGjJD-wPKZGTpLAQKdVMX8nzp926F2/view) | `5669303d6b222c45655b69f780dd4cd0eb1f093863217e4eb9e7195807bfd895` |
| holdout_BigGAN_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/18c7JODcChpFLFEda2XY-BFVebRW86uPx/view) | `918acc2e6a77dd6f24db60c9961cf1035f7d1cf0240767df8143df1d168603b9` |
| holdout_BigGAN_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1y659VH5avv9KQe-Pid1cCXhy31QjZbzm/view) | `6f37ff909ca1b2ac80ad7daec23927436d3523317c128986a2effb72dc92998c` |
| holdout_glide_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/12noaRq0OlH1BbzNP-l4kJ-8O2fqiQRC4/view) | `c3d07c604c262340e2396117e3d58ab50959ba29bf02db3a5d6f769b4005400c` |
| holdout_glide_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1bHJc3ClzIIFqEm-67zqSNmpTrKvVMk4t/view) | `6d4a7418b5c522c34edc95e97b7759440addd9098582999954dd994c2e169097` |
| holdout_Midjourney_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/14qKeiHfOgoteBWZTzsJ5kePT0wIVgV8l/view) | `a520b20274c615f3fd6623607242eaedcedfdf60f4d8f310358c8cdbd225c311` |
| holdout_Midjourney_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1NEanXeZpdhP7psKG4jSz8FVmn7MFgcD3/view) | `9193be3990f4fcfb94a0c95098e7d8e6421668bb31b8b24e3a493f56fc5a1950` |
| holdout_stable_diffusion_v_1_4_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/1Pr05-9SHIWwGvQAJnHZ-F3LxVsg01vZD/view) | `660b61b58ff79750464b0dd3390bec89590339f2dde7fcd94794049a6789fb62` |
| holdout_stable_diffusion_v_1_4_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1GZErUQH6nxdDUNbJzXSxUnIhDrTKto1S/view) | `68c83fc659903454bbf9d225c824617800d2e1a4116790aa52468034b94b5bf6` |
| holdout_stable_diffusion_v_1_5_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/1ddMydDRwUYilhMqb39occ0mJdj9JpXXt/view) | `3f452951df144b1c5af9c5fe635a07d0263bf6b16e1ca6deb5b4ee795d3dd4c2` |
| holdout_stable_diffusion_v_1_5_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1xqrrk_CjiqfIZY2TSuLou8_E6tYQ9rOv/view) | `754792d1f09792a62286f8a1a26f6ff2fbaabd6b922133292eddcb256757cbd8` |
| holdout_VQDM_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/1TC2QEQWL1iixkN6so_FSPVtSVbmiN6Cz/view) | `a83f60138e22b7eb4bb6777feb50e2316b987cbc2b91f8fdd30af2702e7b3302` |
| holdout_VQDM_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1FQzCfQ_E7pYwxacaqk2kwwGl3X5dmsTa/view) | `3fa52f178574d2e4c50107fe7d92b34222350cf9f0c8f4441b9e3173f0528aff` |
| holdout_wukong_k7_rep0__augmented__seed17 | [file](https://drive.google.com/file/d/1t9zBa6SwFKD_GpBoE1htDZfgJXyvK-i7/view) | `320096ab6f6552beb92a4d8a715b07a94b93c60dde4cef63cfb95bc475b5dc2f` |
| holdout_wukong_k7_rep0__baseline__seed17 | [file](https://drive.google.com/file/d/1E1tw_-PgaHNuigUOmAWy16i0ZzbM-n0k/view) | `44ac455dd50e32ee2b4a431adb4924cbd9ec5cc520372b6132bf08d569edf6c6` |
