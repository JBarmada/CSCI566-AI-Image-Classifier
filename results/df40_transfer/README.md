# DF40 transfer evaluation

This folder contains the team's exploratory Celeb-DF and FaceForensics++
results for existing GenImage-trained CNN and CLIP ViT models. Start with the
[experiment log](FINAL_REPORT_EXPERIMENT_LOG.md) for methods, results,
limitations, and the history of decisions. Historical entries describe the
state at the time; later entries record subsequent work.

Current frontier/API progress and maintained commands are in the
[LLM evaluation guide](../../docs/llm-evaluation.md). The 189-image OpenAI run
and matched six-model comparison passed their audits; the
[local exports and analysis](openai_face/df40_five_method_v3_v2/README.md) are available.
The [cross-branch consolidation plan](../../docs/results-consolidation.md)
also inventories the separate Claude/GPT GenImage outputs and extended ViT
confidence analysis. They await integration and should not be confused with
the face results or treated as one pooled benchmark.

- [Celeb-DF evaluation and per-image predictions](shared_results_v1/README.md)
- [DeepFakeFace still-photo CNN–ViT evaluation](still_face_v1/README.md)
- [Still-photo geometry and JPEG follow-up](still_face_geometry_v1/README.md)
- [Still-image dataset comparison and real-error research](STILL_FACE_RESEARCH.md)
- [Paired CNN–ViT error analysis](paired_analysis/README.md)
- [Zero-shot CLIP prompts and results](zeroshot/README.md)
- [FaceForensics++ frozen thresholds and test report](new_data/ff_results_v1/test_report.json)
- [FaceForensics++ zero-shot CLIP test report](new_data/ff_zeroshot_results/test_report.json)
- [FaceForensics++ model comparison](new_data/ff_model_comparison.csv)
- [Supervised face adaptation protocol and recovered results](new_data/ADAPTATION_PROTOCOL.md)
- [Matched clean CNN–ViT confidence audit](clean_cnn_audit/README.md)
- [Audited eight-holdout CNN–ViT robustness comparison](cnn_robustness_v100/README.md)
- [Earlier P100 robustness results and replay history](cnn_robustness_extension/README.md)
- [Open vision-language model comparison](multimodal_qwen/README.md)
- [Paired review of objective comparisons](adaptation_paired_review/README.md)
- [Eight-holdout, three-seed objective comparison](generator_objectives_seeds/README.md)
- [First-seed eight-holdout objective comparison](generator_objectives_extended/README.md)
- [Original ADM objective pilot](generator_objectives/README.md)
- [Draft poster story and figures](poster_assets/README.md)
- [Face-transfer and adaptation follow-up decisions](FACE_FOLLOWUP_DECISIONS.md)
- [Team-review report draft](FINAL_REPORT_DRAFT.md)
- [Team handoff and demonstration](TEAM_HANDOFF.md)
- [FaceForensics++ validation and test manifests](new_data/ff_complete_v1/)
- [Team notebook audit](notebook_audit.md)
- [Threshold calibration protocol](calibration_protocol.md)

The 500-image Celeb-DF subset is available in the [team Drive folder](https://drive.google.com/drive/folders/1OAKFpoLvSQRddGqXgKPek6b7cNHTgb1-).
The FaceForensics++ subset is staged on CARC at
`/project2/yzhao010_1531/csci566-genimage/data/df40_ff_v1/`. These are
external data locations and require appropriate access. Image data, contact
sheets, image archives, and the large pretrained backbone are not included in
Git. The small adapted linear head is included. Manifests retain source paths
and SHA-256 values for verification.

Some scripts contain the original user's local or CARC paths and must be
adapted to the environment before reuse. The saved results are preliminary;
the FaceForensics++ partitions separate listed video IDs, while person-level
and cross-dataset independence are not established.
