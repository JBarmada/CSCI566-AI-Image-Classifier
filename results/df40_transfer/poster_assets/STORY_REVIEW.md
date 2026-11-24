# Poster story: evidence for discussion

**Draft for Jad and team review, November 23, 2026.**

## Research question

When an image detector faces unfamiliar generators, image processing and
manipulated faces, which apparent gains survive a closer look at the metric and
the kind of shift?

The project connects three findings rather than claiming that generator
diversity itself is a new discovery:

1. More training generators are associated with higher clean held-out-generator
   accuracy at the fixed image budget, but the heatmap in the submitted
   midterm shows generator-specific exceptions.
2. Augmentation changes clean, JPEG and resize results differently. For JPEG,
   ViT accuracy increases from 69.24% to 75.99%, while mean AUROC changes from
   0.84987 to 0.84941. The [paired error analysis](JPEG_ERROR_ANALYSIS.md)
   shows the tradeoff: at threshold 0.5, real false positives fell from 54.11%
   to 17.02%, while fake detection fell from 92.59% to 69.00%. Accuracy is
   measured at a threshold; AUROC summarizes ranking across thresholds. Similar
   mean AUROC does not imply identical rankings or prove a calibration mechanism.
3. In exploratory face-transfer sets, false alarms on real images and missed
   fakes depend on the dataset and manipulation. This motivates testing both
   error types, rather than treating accuracy as the entire story. The sets
   differ in multiple ways and do not identify a causal explanation.

## Figures to discuss

- Revisit the submitted midterm's per-generator heatmap and processing examples;
  they show heterogeneity and the intervention more clearly than a macro-only
  plot. Confirm image rights and legibility before reuse.
- [Accuracy and AUROC changes](04_accuracy_auroc_tradeoffs.svg): each dot is
  one held-out generator; the diamond weights the eight holdouts equally.
  Rows compare separately trained baseline and augmented models. The plotted
  differences are descriptive, not independent confidence intervals. Consider
  a simplified single-panel JPEG version if this is too dense for the poster.
- [Simplified ViT JPEG panel](05_vit_jpeg_accuracy_auroc.svg): a poster-sized
  candidate that shows the change in each metric directly. Labels give both
  baseline and augmented values; the common horizontal axis is in percentage
  points (AUROC is multiplied by 100). There are no uncertainty intervals.
- [Class-specific JPEG tradeoff](06_vit_jpeg_tradeoff.svg): shows the decrease
  in real false alarms alongside the decrease in detected fakes. A stronger
  candidate for the main poster than accuracy alone, subject to team review.
- [Existing real-image error plot](03_real_image_errors.svg): include fake
  recall alongside it and prominently label the different datasets. Do not
  infer camera effects or image processing as the cause from this comparison.
- The CNN and ViT have different pretraining and training procedures, so the
  comparison does not isolate architecture. Face identity overlap and the fake
  mirror's byte equivalence remain unverified.

## Context and design guidance

Park and Owens already found that increasing generator diversity can improve
generalization with a fixed image budget; their study is much larger, so our
diversity finding is a replication/context, not a claim of novelty ([paper](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html)).
Grommelt and colleagues show that JPEG and size biases can affect GenImage
detectors ([paper](https://arxiv.org/abs/2403.17608)); this motivates, but does
not establish, an explanation for our real-face errors.

For the actual poster, use direct labels and short, descriptive figure titles;
ensure figure text remains readable at printed size and distinguish series by
shape or line style as well as color. These are design recommendations, not
course rules ([NIH](https://www.training.nih.gov/creating-a-scientific-poster/),
[CMU](https://guides.library.cmu.edu/scienceposters/design),
[LSU](https://libguides.lsuhsc.edu/posterprinting/Design)).

## Next evidence checks

Investigate a predeclared fixed-FPR comparison using separate validation data;
the existing score analysis does not establish a calibrated operating point.
Inspect video and identity provenance before calling any new face test
independent. Select the final story and figures with Jad and the team.
