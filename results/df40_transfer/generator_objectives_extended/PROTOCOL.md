# Eight-holdout extension: frozen feature objectives

Before running this extension, fix the three objectives, hyperparameters, seed,
checkpoint selection, and test threshold exactly as in the ADM pilot described
in `../generator_objectives/PROTOCOL.md`. Add all seven remaining holdouts,
each using its k=7 repeat-0 baseline split. Do not tune against test results.
Use existing CLIP features only and require the original saved linear head to
reproduce validation and test scores before training. Independently replay each
selected head and its validation AUROC and test predictions. Every comparison
uses the same samples within its holdout. Treat all results as exploratory:
the original tests were examined already, and the extension was decided after
seeing the ADM pilot. One repeat and one optimization seed per holdout cannot
measure seed variability. A later untouched evaluation is needed.
