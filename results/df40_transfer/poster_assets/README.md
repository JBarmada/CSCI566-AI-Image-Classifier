# Poster story and figure selection

**Owner: Jad. Status: recommended for team review, not a finished poster.**

## One-sentence story

At a fixed image budget, training on more generator families was associated
with better detection of held-out generators; augmentation improved robustness
to our JPEG and resize tests, while transfer to manipulated faces exposed
important real-image and detection errors.

The CNN is a fine-tuned ResNet-50; the ViT is a frozen CLIP ViT-L/14 encoder
with a trained linear head. They differ in pretraining and training, so their
comparison cannot isolate architecture. All k=7 results average repeats within
each of eight held-out generators and then weight the holdouts equally. The
same test images recur; do not interpret repeated rows as independent samples.

## Recommended primary figures

1. **Generator diversity:** [PNG](01_generator_diversity.png) · [SVG](01_generator_diversity.svg). Clean accuracy versus the number of training generators, with baseline and augmented CNN and ViT. This answers the project's main question. Sources: [CNN summary](../clean_cnn_audit/equal_weight_holdout_summary.csv) and [ViT summary](../../vit_benchmark/summary/macro_across_holdouts.csv). The y-axis begins at 50%; label it clearly and avoid implying uncertainty intervals.
2. **Robustness:** [PNG](02_robustness.png) · [SVG](02_robustness.svg). Clean, JPEG60 and resize accuracy at k=7, comparing baseline and augmented pipelines. Clean accuracy is lower with augmentation, while JPEG and resize accuracy are higher in these averages. Source: [audited V100/P100 comparison](../cnn_robustness_v100/paired_macro.csv).
3. **Real-image errors:** [PNG](03_real_image_errors.png) · [SVG](03_real_image_errors.svg). At the original 0.5 threshold, baseline CNN/ViT falsely flagged 99/100 and 90/100 DF40 real video crops, versus 23/50 and 4/50 real DeepFakeFace photos. Sources: [DF40 metrics](../shared_results_v1/metrics.csv) and [still-photo metrics](../still_face_v1/metrics.csv). **Required caption:** these are small, different datasets with different origins and processing; they do not isolate the cause of errors. This figure does not show fake recall, so pair it with fake-detection results or the caption below.

Suggested caption for figure 3: “False positives on real images at threshold 0.5. The sets differ in source, processing, and size; these descriptive results do not isolate a cause. In the DF40 test, fake detection and AUROC varied by manipulation. The same 100 real images were reused across methods.”

## Other results: keep exploratory

- **Face adaptation:** optional small callout, not a main evidence panel. The supervised head trained on all three evaluated methods and its test was previously inspected; video-ID separation does not establish person independence.
- **Adversarial and contrastive objectives:** optional small table or discussion. Differences were small and the extension was selected after earlier test results; additional seeds used the same dataset repeat.
- **Qwen and zero-shot CLIP:** use only if the team includes the relevant teammate's work and has room to explain its protocol. One small Qwen pilot does not represent a broad frontier-model comparison.

Generate the three figures from the saved CSV files with `build_figures.py`;
PNG and editable vector SVG versions are provided. Figures are drafts for team
review. No data were selected based on test outcomes for these plots beyond the
clearly labeled, already explored evaluations.
