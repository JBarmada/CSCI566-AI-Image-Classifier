# Poster handoff: draft for team review

On November 27, Jad approved the overarching research question:

> **How well do AI-image detectors generalize when the generator, image
> presentation, and type of manipulation change?**

This question connects the original proposal to the post-midterm experiments.
A complete [review-ready poster PDF and SVG](../../../docs/final_delivery/README.md), new source-linked figures, report and browser-verified demo are now available. Team review of the final layout and captions remains open.

## Proposed poster structure

1. **Generator change.** Show the matched clean CNN–ViT results as training
   generator diversity increases at a fixed image budget. Explain the shared
   holdouts and the differences between the model pipelines.
2. **Image presentation change.** Show the saved-checkpoint robustness results
   and/or the matched GPT/Claude image-view comparison. The paired 590-source
   comparison holds the image IDs fixed across original, square-cropped and
   detector224 views. Consider the calibration findings as supporting evidence.
3. **Manipulation change.** Show how the same detectors respond to real and
   manipulated face crops. Use the verified 189-image comparison, separate
   real false alarms from fake recall, and label the face-adapted head as
   supervised. Keep provenance and identity limitations visible.

Use the [independent readiness audit](../../llm_eval/team_readiness_2026-11-26/README.md)
for the latest matched results, coverage counts and scientific limitations.
The JPEG error tradeoff and earlier story options below remain available as
supporting results.

## Supporting robustness result: JPEG error tradeoff

**Question:** When detector accuracy improves on compressed images, what
happens to false alarms on real images and missed fake images?

**Candidate figure:** [JPEG tradeoff](06_vit_jpeg_tradeoff.svg)
([PNG](06_vit_jpeg_tradeoff.png)). For the frozen CLIP ViT-L/14 pipeline at
threshold 0.5 on JPEG quality 60 images, the augmented model flags fewer real
images as fake (54.1% → 17.0%) but detects fewer fake images (92.6% → 69.0%).
The figures are mean rates across eight equally weighted held-out generators,
after averaging three dataset repeats. Overall accuracy rises from 69.24% to
75.99%; mean AUROC is approximately 0.85 in both cases.

**Suggested caption:** “At the original decision threshold, the augmented
ViT makes fewer false alarms on real JPEG images but misses more fake images.
Eight generator holdouts are weighted equally; repeated test images are not
independent observations.” [Methods and calculations](JPEG_ERROR_ANALYSIS.md).

The figure shows a comparison of separately trained models at one threshold.
Similar average AUROC does not mean identical rankings, prove why scores
changed, or show that the same tradeoff holds at every threshold. A descriptive
10% false-positive test-ROC comparison uses the test labels to choose points;
it is not an independently validated deployment threshold.

For discussion: both rates fall in all eight holdout means, though the fake
recall decrease ranges from 17.02 to 41.55 percentage points. See the
[generator-level breakdown](JPEG_ERROR_ANALYSIS.md) and the optional
[per-generator figure](09_jpeg_holdout_changes.svg)
([PNG](09_jpeg_holdout_changes.png)). This figure only shows fake recall, so
pair it with the main tradeoff panel if used. The results are descriptive;
the holdouts share some real-image test data.

The CNN shows the same direction on JPEG60 at its original 0.5 threshold:
real false alarms 65.79% → 20.01%, fake detection 94.64% → 61.32%. See the
[audited per-image analysis](JPEG_ERROR_ANALYSIS.md#does-the-cnn-show-the-same-direction).
An alternative [two-model figure](10_cnn_vit_jpeg_tradeoff.svg)
([PNG](10_cnn_vit_jpeg_tradeoff.png)) places both tradeoffs side by side for
Jad and team review. This is a whole-pipeline comparison, not an
architecture-only result.

## Supporting figures to discuss

1. **Where does performance vary?** The [generator heatmap](07_vit_generator_heatmap.svg)
   shows clean baseline ViT accuracy by held-out generator and training
   diversity. It adds detail to the overall average. Training subsets differ
   across diversity settings, so a decline does not identify the effect of
   adding a particular generator. Prior research has also studied generator
   diversity; see [literature context](STORY_REVIEW.md#context-and-design-guidance).
2. **Do class-specific errors matter on faces too?** The exploratory
   [face-error figure](08_face_error_tradeoff.svg) places false alarms next to
   fake detection. The two face datasets differ in source, processing and
   manipulation; do not attribute their differences to cameras or a single
   cause. Identity separation and some fake provenance remain unverified.

These are candidate supporting panels, subject to space and team review. A
CNN–ViT comparison describes whole pipelines: their pretraining and training
also differ, so it cannot isolate architecture. The five-method follow-up
passed replay checks on P100 but remains exploratory; review the
[results and data limitations](../new_data/fresh_candidate_results_v3/README.md)
before deciding whether it belongs on the poster.

## Presenter explanation

Accuracy counts correct decisions at a chosen threshold. AUROC summarizes how
well scores rank real and fake images across thresholds. A model can therefore
have higher accuracy at one threshold while missing more fakes, especially when
it also makes fewer false alarms. Our balanced benchmark supports the numerical
comparison above; real-world class proportions can change the practical impact.

Before finalizing, verify all labels and citations against the linked evidence,
check readability at the intended print size, and decide with Jad and the team
which supporting panel best serves the space available.
