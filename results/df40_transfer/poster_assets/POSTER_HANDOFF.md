# Poster handoff: draft for team review

Jad selected the JPEG error tradeoff as the lead. This is a proposed story and
figure order for the team, not an approved poster or a claim of a new general
discovery. The [full story review](STORY_REVIEW.md) includes alternatives.

## Question and main result

**Question:** When detector accuracy improves on compressed images, what
happens to false alarms on real images and missed fake images?

**Main figure:** [JPEG tradeoff](06_vit_jpeg_tradeoff.svg)
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
also differ, so it cannot isolate architecture. The five-method follow-up is
queued and must not be shown as a result until it completes and passes checks.

## Presenter explanation

Accuracy counts correct decisions at a chosen threshold. AUROC summarizes how
well scores rank real and fake images across thresholds. A model can therefore
have higher accuracy at one threshold while missing more fakes, especially when
it also makes fewer false alarms. Our balanced benchmark supports the numerical
comparison above; real-world class proportions can change the practical impact.

Before finalizing, verify all labels and citations against the linked evidence,
check readability at the intended print size, and decide with Jad and the team
which supporting panel best serves the space available.
