# DF40 transfer evaluation

This folder contains the team's exploratory Celeb-DF and FaceForensics++
results for existing GenImage-trained CNN and CLIP ViT models. Start with the
[experiment log](FINAL_REPORT_EXPERIMENT_LOG.md) for methods, results,
limitations, and the history of decisions. Historical entries describe the
state at the time; later entries record subsequent work.

- [Celeb-DF evaluation and per-image predictions](shared_results_v1/README.md)
- [Paired CNN–ViT error analysis](paired_analysis/README.md)
- [Zero-shot CLIP prompts and results](zeroshot/README.md)
- [FaceForensics++ frozen thresholds and test report](new_data/ff_results_v1/test_report.json)
- [FaceForensics++ validation and test manifests](new_data/ff_complete_v1/)
- [Team notebook audit](notebook_audit.md)
- [Threshold calibration protocol](calibration_protocol.md)

The 500-image Celeb-DF subset is available in the [team Drive folder](https://drive.google.com/drive/folders/1OAKFpoLvSQRddGqXgKPek6b7cNHTgb1-).
The FaceForensics++ subset is staged on CARC at
`/project2/yzhao010_1531/csci566-genimage/data/df40_ff_v1/`. These are
external data locations and require appropriate access. Image data, contact
sheets, archives, and model weights are not included in Git. Manifests retain
source paths and SHA-256 values for verification.

Some scripts contain the original user's local or CARC paths and must be
adapted to the environment before reuse. The saved results are preliminary;
the FaceForensics++ partitions separate listed video IDs, while person-level
and cross-dataset independence are not established.
