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

- [x] **Update the main README and final report draft.** **Owner: Jad.**
      Corrected the stale replay status and linked the [audited comparison](results/df40_transfer/cnn_robustness_v100/README.md).
- [ ] **Choose the main story and figures.** **Owner: Jad.** Cover generator diversity, the
      CNN–ViT comparison, JPEG/resize robustness and transfer to manipulated
      faces. A [draft story and three figures](results/df40_transfer/poster_assets/README.md)
      are ready for team review; choose which exploratory results fit the poster.
- [ ] **Review the three linked research questions.** **Owner: Jad.** Compare
      threshold accuracy with AUROC, examine real-image false positives alongside
      missed fakes, and show variation across held-out generators. See the
      [evidence and new draft figure](results/df40_transfer/poster_assets/STORY_REVIEW.md).
      [JPEG error breakdown](results/df40_transfer/poster_assets/JPEG_ERROR_ANALYSIS.md)
      shows fewer real false alarms but also fewer detected fakes; investigate
      a descriptive fixed-FPR ROC comparison is available; select an operating
      threshold on separate validation data before deployment claims.
- [ ] **Create the poster and prepare the demo.** **Owners: Jeslyn and Rachel.** Explain the question, methods,
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
- [x] **Document face-data limitations.** **Owner: Jad.** See the report draft
      and [follow-up decisions](results/df40_transfer/FACE_FOLLOWUP_DECISIONS.md).
      Person-level independence is unverified and the adapted head used a
      previously inspected test set.
- [ ] **Check citations and reproducibility.** Verify references, dataset
      provenance, manifests, model settings, and team access to GitHub, CARC and
      the shared data. Confirm all reported figures can be regenerated.
- [ ] **Confirm course submission requirements.** Check the latest Piazza and
      Gradescope instructions, report format, poster logistics and peer
      evaluation; assign owners and confirm deadlines with the team.

## Research choices if time permits

- [ ] **Decide whether to collect a fresh face test set.** **Owner: Jad.** A test with verified
      source-video and person separation would strengthen transfer claims.
      Keep any calibration data separate from that test. [Recommendation and
      pilot protocol](results/df40_transfer/FACE_FOLLOWUP_DECISIONS.md) are ready
      for team review; no validated independent test has yet been assembled.
      Working decision: prepare one if provenance checks pass; an initial
      [candidate-group inventory](results/df40_transfer/FACE_FOLLOWUP_DECISIONS.md)
      finds possible unused video IDs, but cannot verify person independence.
      A [203-image candidate](results/df40_transfer/new_data/FRESH_CANDIDATE_REVIEW.md)
      is on CARC. One pair was excluded after cross-method duplicate bytes;
      provenance limitations and the frozen exploratory protocol are recorded.
      Inference job 14525004 is queued; its results are not yet available.
- [ ] **Decide whether to include more manipulation methods.** **Owner: Jad.** The reported
      DF40 tests cover SimSwap, FOMM and Wav2Lip. Add methods only with
      documented provenance, sampling and comparable evaluation. A limited
      extension is [recommended](results/df40_transfer/FACE_FOLLOWUP_DECISIONS.md)
      if access, verification and time allow.
      Working decision: InSwapper and PIRender archive indexes match their
      metadata copies. Check image bytes, source overlap, visible manipulations
      and access terms before constructing the shared evaluation.
      Selected candidate images for both methods exist, but a duplicate pair
      across two method archives requires further label-provenance review.
- [ ] **Strengthen adaptation comparisons if useful.** **Owner: Jad.** More independent data
      splits and a fresh test would help assess the exploratory adversarial and
      contrastive results; their present small differences do not establish a
      reliable improvement. See the [separate face and generator-objective
      plans](results/df40_transfer/FACE_FOLLOWUP_DECISIONS.md) and [paired
      review of existing runs](results/df40_transfer/adaptation_paired_review/README.md).
      Working decision: compare the frozen face-adapted head on the fresh shared
      test first; defer new training until the test and protocol are protected.
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

**Recommended next step:** review the draft poster figures and decide whether
the team can assemble a fresh, traceable face test before the deadlines.
