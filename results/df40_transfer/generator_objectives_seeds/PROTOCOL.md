# Additional optimizer seeds for objective comparison

After seeing the seed-17 eight-holdout exploratory results, run the identical
binary, adversarial and contrastive feature adapters at optimizer/loader seeds
29 and 43 for every holdout. Keep the fixed repeat-0 manifests, feature caches,
hyperparameters, validation AUROC selection and test threshold unchanged.
The CLIP backbone stays frozen. Verify the old linear-head scores and replay
every resulting checkpoint. Compare seed-averaged holdout metrics and report
all outcomes; this is a post-hoc exploration, not a pristine test. These seeds
measure optimizer variability, not dataset-split variability, and test images
are reused. No hyperparameter or objective is selected using these results.
