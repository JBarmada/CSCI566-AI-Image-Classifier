# Poster story: evidence for discussion

**Jad selected direction A as the lead on November 23, 2026.** Directions B
and C remain supporting candidates; the poster's layout and final wording are
still for team review.

## Selected lead

On compressed images, an augmented ViT achieved higher accuracy by flagging
fewer real images as fake, but it also missed more fakes. The tradeoff is
visible at the existing 0.5 threshold; mean AUROC and test-derived fixed-FPR
results show why that single threshold does not describe the whole detector.
The result is specific to the evaluated models, JPEG60 images and benchmark.

## Three directions for the main story

**A. A robustness gain with a hidden cost.** On JPEG60 images, the augmented
ViT's accuracy improved at threshold 0.5 because its false alarms fell more
than its fake recall did. Mean AUROC barely changed. This direction connects
the measured accuracy, decision threshold, class-specific errors and model
ranking. The same-test fixed-FPR analysis is descriptive, not calibration.
[Tradeoff figure](06_vit_jpeg_tradeoff.svg) ·
[evidence](JPEG_ERROR_ANALYSIS.md).

**B. A detector can appear to find nearly every fake by flagging nearly
everything.** On the exploratory DF40 SimSwap sample, baseline CNN detected
199/200 fakes but falsely flagged 99/100 real crops at threshold 0.5. Its
200/300 correct predictions equal the accuracy of an always-fake classifier on
that class-imbalanced sample. Baseline ViT similarly flagged 90/100 reals.
The different DeepFakeFace photo sample has different results, but data source,
processing and manipulation also differ, so the comparison does not identify
the cause. [Face error figure](08_face_error_tradeoff.svg) ·
[video metrics](../shared_results_v1/metrics.csv) ·
[photo metrics](../still_face_v1/metrics.csv).

**C. Diversity helps on average, but results depend on the held-out
generator.** At a fixed image budget, the baseline ViT average improves from
k=1 to k=7, yet seven of eight generator rows contain at least one decline
between consecutive k settings. Different training subsets are used across
k; this is variation in the observations, not proof that adding a particular
generator harms performance. The broader claim that generator diversity helps
has prior work. [Generator heatmap](07_vit_generator_heatmap.svg) ·
[source table](../../vit_benchmark/summary/by_holdout_k.csv).

Jad selected A as the lead. B and C can support the story if the poster has
space; each needs a clear role and caveats. The team has not chosen the layout.

## Research question

When an image detector faces unfamiliar generators, image processing and
manipulated faces, which apparent gains survive a closer look at the metric and
the kind of shift?

The project connects three findings rather than claiming that generator
diversity itself is a new discovery:

1. More training generators are associated with higher clean held-out-generator
   accuracy at the fixed image budget, but results vary by held-out generator.
   In baseline ViT means, seven of eight rows have at least one decrease
   between consecutive k settings. Different training subsets are used at
   different k, so these dips do not establish the effect of adding a specific
   generator. See the [recreated heatmap](07_vit_generator_heatmap.svg).
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
- [Recreated midterm-style heatmap](07_vit_generator_heatmap.svg): baseline
  ViT clean accuracy, averaging three repeats per cell; fixed image budget.
  Its numerical labels and color scale show where the macro trend varies.
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
- [CNN and ViT JPEG tradeoffs](10_cnn_vit_jpeg_tradeoff.svg): puts both
  pipelines' class-specific changes side by side as an alternative lead panel.
- [JPEG fake-detection change by holdout](09_jpeg_holdout_changes.svg): shows
  how the fake-recall decrease varies by generator; pair with the class-specific
  panel above because this breakdown does not display real false alarms.
- [Existing real-image error plot](03_real_image_errors.svg): include fake
  recall alongside it and prominently label the different datasets. Do not
  infer camera effects or image processing as the cause from this comparison.
- [Face false alarms and fake detections](08_face_error_tradeoff.svg): pairs
  both error types for baseline CNN and ViT on two small face samples. The
  dataset and manipulation differences must stay visible in the caption.
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
Dolhansky and colleagues' 2019 DFDC preview also emphasize the importance of
false positives and differences between benchmark and real-world class balance
([paper](https://doi.org/10.48550/arXiv.1910.08854)). Our contribution is the
specific measured tradeoff in these experiments, not the general observation
that false positives or class balance matter. Scite returned restricted full
text for some other search results; no claims here rely on those papers.

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
