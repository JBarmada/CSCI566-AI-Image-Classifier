# Controlled face adaptation

This experiment trains one new linear classifier on frozen OpenAI CLIP
ViT-L/14 image features. The baseline is the previously trained GenImage
head and the fixed-prompt zero-shot CLIP result on the same FF test subset.
The previously inspected FF test results make this a follow-up experiment,
not a pristine confirmatory test.

The frozen manifest selects one crop per group from DF40 FF training metadata.
Training has 120 real and 120 fake images (40 per method). Validation has 60
real and 60 fake images (20 per method). Source IDs were partitioned before
sampling and known IDs from the 140-video FF test pool were excluded. In the
SimSwap training metadata some groups reference test IDs, so those were
explicitly removed. Related or renamed clips and person identity overlap are
not fully resolved. Source archive provenance and image hashes are recorded in
the manifest; the fake crop mirror has not been proven byte-identical to the
official archive.

The frozen encoder uses original CLIP preprocessing and normalized float
features. Train a randomly initialized linear head with seed 566, AdamW
learning rate 0.001, weight decay 0.0001, batch 32, at most 100 epochs and
patience 15. Select the best validation AUROC, then set the lowest threshold
with at most 10% false positives on the validation real images. Save the head,
epoch, threshold and validation history before reading test images. Report
per-method AUROC, false positives and fake recall at both the fixed 0.5 and
validation-selected thresholds. This is supervised face-domain adaptation,
not domain-adversarial or contrastive training.
