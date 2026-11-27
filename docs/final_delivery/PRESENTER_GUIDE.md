# Presenter guide

## Two-minute story

**0:00–0:20.** Our question is how well AI-image detectors generalize as
the generator, image presentation and manipulation change. We train two
different detection pipelines on fixed-size subsets of GenImage and compare
their saved models, including zero-shot multimodal models.

**0:20–0:50.** The first panel varies the number of training generators.
We hold the image budget constant and test on a generator excluded from
training. Across 192 matched pairs, clean baseline accuracy rises from
62.75% to 83.30% in CNN and 73.00% to 88.30% in ViT. The two pipelines use
different pretraining and training, so do not call this an architecture-only
comparison. The matched frontier studies use smaller source intersections
and their own input preparation.

**0:50–1:20.** The second panel asks whether that ability survives changes in
presentation. JPEG60 augmentation reduces false alarms but also reduces fake
recall at threshold 0.5. Resize accuracy improves. Even GPT/Claude change
decisions on the exact same sources: detector224 reduces fake recall compared
with the original image view. The intervention also involves crop/resize and
not identical internal tensors; we cannot isolate resolution alone.

**1:20–1:50.** The third panel tests a different task: manipulated faces.
Both original detectors and zero-shot models exhibit serious errors. CNN
flags 53/54 real crops; GPT detects only 16/135 fakes. Their balanced accuracy
is near 50% for opposite reasons. The adapted ViT reference is promising
but was supervised and the test was previously examined; identity and mirror
label provenance remain unverified.

**1:50–2:00.** The takeaway is that good performance on unseen generators
does not guarantee robustness to presentation or transfer to localized
manipulations. Report both errors and ranking, qualify the test distribution,
and use a fresh, independently verified test before deployment claims.

## Offline demo (2–3 minutes)

1. Download and open [demo.html](demo.html) in a browser (no network needed).
   The clean table shows the fixed-budget experiment.
2. In GenImage presentation study, compare GPT/Claude original vs square vs
   detector224. The threshold starts at 50%; check 284/520 vs 142/520 GPT
   fake detections, and 388/520 vs 291/520 Claude.
3. Switch to DF40 faces. Contrast CNN baseline 53/54 false alarms and
   133/135 fakes detected with GPT 6/54 and 16/135. Same 189 source images.
4. Select one source: compare its saved scores with its recorded label.
   Move the threshold briefly to illustrate the tradeoff, then reset.
5. State clearly that this is an offline replay, not new model inference.
   If the live environment is unavailable, use this tested backup and the
   poster PDF. The original datasets are not bundled; do not use restricted
   image crops publicly without checking permissions.

## Likely questions

- **Why balance training counts?** Separate the number of generators from
  the overall training-image count, though source subsets still differ.
- **Why not just accuracy?** A face test has 135 fake and 54 real images;
  always predicting fake achieves 71.4% accuracy and 50% balanced accuracy.
- **Do CNN and ViT differ only in architecture?** No, they differ in
  pretraining, heads/fine-tuning and augmentation.
- **Does data augmentation improve everything?** No, it sacrifices clean
  accuracy and shifts class errors; gains vary with condition and model.
- **What is AUROC?** Ranking of scores over possible thresholds, not a
  guarantee of calibrated confidence or accuracy at threshold 0.5.
- **What explains real face errors?** Face cropping, dataset source and
  acquisition/processing differences are plausible, not established causes.
  A controlled processing follow-up found limited changes in 50 real images;
  camera-specific effects were not measured.
- **Why not a broad frontier leaderboard?** Each comparison has a matched
  subset, provider configuration and fixed prompt; invalid GPT outputs are
  disclosed. GenImage and face GPT are different model IDs.
- **Did adaptation solve face transfer?** No independent person-disjoint
  evaluation exists. The 67.78% balanced accuracy is an exploratory supervised
  reference on the selected set. The separate adversarial/contrastive
  feature-adapter experiments did not establish an improvement.
- **What about confidence?** Validation-fitted temperature scaling improved
  k=7 ViT ECE/log loss but made k=1 log loss worse. Reliability can shift as
  generators shift; avoid quoting the bootstrap CI without fixing dependence.

Sources: [report](../../results/df40_transfer/FINAL_REPORT_DRAFT.md) and
[audit](../../results/llm_eval/team_readiness_2026-11-26/README.md).
