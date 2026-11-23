# Exploratory generator-objective comparison

The frozen CLIP feature adapter compared three predeclared objectives on the
ADM k=7 rep0 GenImage split. The same training and validation data, seed,
optimizer, adapter architecture, and validation-AUROC selection rule were used.
The adversarial objective reverses gradients from a generator classifier trained
only on fake examples. The contrastive objective groups same-class examples in
the feature adapter. The CLIP encoder was frozen in all three runs. See
[protocol](PROTOCOL.md) for settings and limitations.

| Objective | Test accuracy | AUROC | Real false positives | Fakes detected |
| --- | ---: | ---: | ---: | ---: |
| Binary baseline | 0.8893 | 0.9664 | 17/280 | 235/280 |
| Generator-adversarial | 0.9036 | 0.9679 | 18/280 | 244/280 |
| Supervised contrastive | 0.8875 | 0.9676 | 17/280 | 234/280 |

The baseline and two regularized runs share the adapter architecture. They
are not the original linear classifier. The fake-domain classifier used only
the seven training generators; ADM was held out. Feature identities and
labels were checked against the original experiment, and the original linear
head replay differed from its saved scores by at most 1.22e-7 on validation and
9.15e-8 on test. Separate checkpoint replay verified the selected validation
AUROC and reproduced each test score within 3e-8. Saved weights, selection
history and per-image predictions are in `results_v1`.

These are exploratory results for one previously inspected holdout and one
seed, without hyperparameter sweeps or uncertainty estimates. The small
observed differences do not establish that either objective generalizes better.
The baseline and contrastive selected checkpoints were at epoch 20, the
configured training limit. No backbone parameters were adapted.
