# CSCI 566 project backlog

Updated November 23, 2026. The primary clean CNN–ViT benchmark and the V100
CNN robustness rerun are complete. Focus first on communicating verified
results and completing course deliverables; additional experiments are choices,
not prerequisites for reporting the work already done.

The official [course schedule](https://usc-fortis.github.io/CSCI566-F26/schedule.html)
lists the poster and demo session on **November 30** and the final report due
**December 7**. Check Piazza and Gradescope for submission instructions and
any updates. Everyone attending the poster session should be ready to answer
questions about the project.

## Before the poster session

- [ ] **Update the main README and final report draft.** Both still describe the
      earlier CNN replay discrepancy as unresolved. The affected V100 reruns
      passed the original clean replay check; use the [audited comparison](results/df40_transfer/cnn_robustness_v100/README.md).
- [ ] **Choose the main story and figures.** Cover generator diversity, the
      CNN–ViT comparison, JPEG/resize robustness and transfer to manipulated
      faces. Decide which exploratory results fit the poster.
- [ ] **Create the poster and prepare the demo.** Explain the question, methods,
      results, limitations and conclusion. Test the demo and prepare a backup
      that does not depend on a live CARC or internet connection.
- [ ] **Practice as a team.** Make sure each member can explain the models,
      findings, limitations and their own contribution.

## Before the final report

- [ ] **Finish the report draft.** Integrate the audited robustness results and
      the team's verified contributions.
- [ ] **Verify every figure, table and number** against saved results, including
      sample counts, decision thresholds and aggregation methods.
- [ ] **Explain the model comparison fairly.** CNN and ViT differ in pretraining
      and training as well as architecture. Repeated test images and dataset
      repeats are not independent samples.
- [ ] **Explain errors on real images.** Report what our processing tests showed;
      treat camera, source and other differences as hypotheses unless directly
      supported by suitable metadata and controls.
- [ ] **Document face-data limitations.** Known FaceForensics++ video IDs are
      separated, but person-level and cross-dataset independence remain
      unverified. The face-adapted model used a previously inspected test set,
      so its findings are exploratory.
- [ ] **Check citations and reproducibility.** Verify references, dataset
      provenance, manifests, model settings, and team access to GitHub, CARC and
      the shared data. Confirm all reported figures can be regenerated.
- [ ] **Confirm course submission requirements.** Check the latest Piazza and
      Gradescope instructions, report format, poster logistics and peer
      evaluation; assign owners and confirm deadlines with the team.

## Research choices if time permits

- [ ] **Decide whether to collect a fresh face test set.** A test with verified
      source-video and person separation would strengthen transfer claims.
      Keep any calibration data separate from that test.
- [ ] **Decide whether to include more manipulation methods.** The reported
      DF40 tests cover SimSwap, FOMM and Wav2Lip. Add methods only with
      documented provenance, sampling and comparable evaluation.
- [ ] **Strengthen adaptation comparisons if useful.** More independent data
      splits and a fresh test would help assess the exploratory adversarial and
      contrastive results; their present small differences do not establish a
      reliable improvement.
- [ ] **Coordinate the frontier-model comparison with its owner.** Include the
      teammate's verified results if ready; that work remains theirs.

## Completed and available

- Corrected clean CNN and ViT benchmarks and the 192-pair comparison.
- Eight-holdout ViT JPEG/resize robustness and the corresponding CNN rerun.
  All 24 affected CNN clean runs matched the original scores on V100; the
  updated 144-row matched comparison passed its replay checks. Earlier P100
  results remain available for provenance.
- Exploratory DF40 face-transfer, FaceForensics++ calibration/adaptation,
  still-photo, paired-error and generator-objective analyses. Their remaining
  data and generalization limitations are recorded in the
  [experiment log](results/df40_transfer/FINAL_REPORT_EXPERIMENT_LOG.md).

**Recommended first step:** update the outdated README and report draft with
the V100 results, then select the poster figures.
