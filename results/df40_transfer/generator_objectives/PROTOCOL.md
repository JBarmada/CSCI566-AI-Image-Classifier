# Exploratory generator-objective comparison

Use the audited GenImage ADM k=7 repeat-0 split, frozen CLIP ViT-L/14 features,
and seed 17. Compare one common 768→128 ReLU feature adapter and binary head
under three objectives: binary classification alone; classification with a
gradient-reversal generator classifier on fake images; and classification with
supervised contrastive loss using real/fake labels. The generator classifier
sees the seven training generators only. It is never trained on real image
source labels or the held-out ADM generator.

Use identical training and validation samples, optimizer (AdamW learning
rate 0.001, weight decay 0.0001), batch size 128, at most 20 epochs and
validation AUROC checkpoint selection with patience 3. Adversarial and
contrastive losses use weight 0.1; contrastive temperature is 0.1. The held-out
test is evaluated only after selection. Threshold remains 0.5. The existing
linear GenImage head serves as an additional reference; it uses a different
head architecture. These are exploratory single-seed, single-holdout results.
The ADM test has already been examined elsewhere and is not pristine for
confirmatory model development. This study operates on frozen CLIP features;
it does not fine-tune the backbone or claim full end-to-end DANN.

The feature cache identity includes ordered image hashes, encoder, preprocessing,
engine and environment. The run must reproduce the existing linear head's
validation and test probabilities before new training starts. Record source
config, feature cache hashes, model checkpoints, predictions and validation
history. Report performance by objective along with limitations.
