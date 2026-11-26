# CSCI 566 project backlog

Updated November 26, 2026. The primary clean CNN–ViT benchmark and the V100
CNN robustness rerun are complete. Focus first on communicating verified
results and completing course deliverables; additional experiments are choices,
not prerequisites for reporting the work already done.

The next priority is a consolidated analysis of the saved outputs. The
[consolidation plan](docs/results-consolidation.md) records completed code,
branch locations, verification gaps and the order of work. Claude, GPT GenImage
and the extended ViT confidence analysis are available on separate branches;
their availability does not mean they are integrated or fully audited here.

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
      Jad selected the JPEG false-alarm versus missed-fake tradeoff as the lead;
      decide with the team which supporting figures fit. A short
      [poster handoff](results/df40_transfer/poster_assets/POSTER_HANDOFF.md)
      connects the lead figure, candidate supporting panels and limitations.
- [ ] **Review the three linked research questions.** **Owner: Jad.** Compare
      threshold accuracy with AUROC, examine real-image false positives alongside
      missed fakes, and show variation across held-out generators. See the
      [evidence and new draft figure](results/df40_transfer/poster_assets/STORY_REVIEW.md).
      [JPEG error breakdown](results/df40_transfer/poster_assets/JPEG_ERROR_ANALYSIS.md)
      shows fewer real false alarms but also fewer detected fakes. A descriptive
      fixed-FPR ROC comparison is available; select an operating threshold on
      separate validation data before making deployment claims. The locally
      available ADM validation predictions are for clean images, so they do not
      establish a 10% false-positive rate on JPEG images. A JPEG-specific check
      needs separate JPEG-processed validation predictions and a frozen rule.
- [ ] **Integrate the completed branch contributions. Owner: to confirm.**
      Preserve our pushed OpenAI face workflow and results (`1324ffc`), add
      Claude/GPT GenImage outputs from `zero-shot/gpt-genimage` (`7be92ec`,
      already including Claude), and add `analysis/vit-confidence-errors`
      (`ebeba98`). Review conflicts and source provenance before merging.
      `main` still lacks these contributions. See the
      [source inventory](docs/results-consolidation.md).
- [ ] **Finish matched frontier GenImage comparison. Owner: Shriya.** Gemini and
      Groq-Qwen completed the 90-image pilot; Claude scored a 630-image superset
      in three views, and GPT now has 1,844 valid / 1,890 image-view outputs.
      Compare only shared valid images and matching views with the corrected
      CNN/ViT exports. Recompute GPT's pilot comparison: its baseline metrics
      currently use 90 images while GPT has 88 valid matches. Its existing
      ViT intersections are tiny, all-real samples from old ADM exports.
      Obtain the matching CNN predictions from CARC and the GPT attempt log
      from its owner. The separate 2,520-image attempts remain incomplete.
- [x] **Complete frontier face evaluation. Owners: Jeslyn, Aditya.** On November
      26, the cleaned OpenAI runner completed a fresh smoke and all 189 frozen
      images on CARC. Summary and matched CNN/ViT comparison audits passed
      (1,134 prediction rows). See the [current workflow](docs/llm-evaluation.md).
- [x] **Archive and analyze the OpenAI face results. Owner: Aditya.** CARC
      exports are stored locally, regenerated CSVs match exactly, and the
      [analysis](results/df40_transfer/openai_face/df40_five_method_v3_v2/README.md)
      and report draft include class-specific errors and dataset limitations.
      Final poster selection and submitted report integration remain team work.
- [ ] **Create the consolidated result analysis. Owner: Aditya.** Build a
      source/protocol inventory, regenerate matched metrics, and provide one
      analysis entry point with separate clean, robustness, frontier GenImage
      and face sections. Include coverage, class-specific errors, limitations
      and evidence links. Use saved outputs; no new API calls or training are
      required for consolidation. Follow the [plan](docs/results-consolidation.md).
- [ ] **Create the poster and prepare the demo.** **Owners: Jeslyn and Rachel. (probably need more people)** Explain the question, methods,
      results, limitations and conclusion. Test the demo and prepare a backup
      that does not depend on a live CARC or internet connection.
- [ ] **Practice as a team.** Make sure each member can explain the models,
      findings, limitations and their own contribution.

## Before the final report

- [ ] **Finish the report draft.** Integrate the audited robustness results and
      the team's verified contributions.
- [ ] **Verify every figure, table and number** **Owner: Rachel.** against saved results, including
      sample counts, decision thresholds and aggregation methods.
- [ ] **Integrate and reproduce the completed ViT confidence/error analysis.**
      The implementation, figures and tests already exist on
      `analysis/vit-confidence-errors`; do not list them as unimplemented.
      Check their sources and validation-only fitting after integration. An
      equivalent extended CNN recalibration/repeated-error study remains a
      separate optional implementation, not a prerequisite for reporting.
- [ ] **Explain the model comparison fairly.** CNN and ViT differ in pretraining
      and training as well as architecture. Repeated test images and dataset
      repeats are not independent samples.
- [ ] **Explain errors on real images.** **Owner: Rachel.** Report what our processing tests showed;
      treat camera, source and other differences as hypotheses unless directly
      supported by suitable metadata and controls.
- [x] **Document face-data limitations.** **Owner: Jad.** See the report draft
      and [follow-up decisions](results/df40_transfer/FACE_FOLLOWUP_DECISIONS.md).
      Person-level independence is unverified and the adapted head used a
      previously inspected test set.
- [ ] **Check citations and reproducibility.** **Owner: Saisha.** Verify references, dataset
      provenance, manifests, model settings, and team access to GitHub, CARC and
      the shared data. Confirm all reported figures can be regenerated.
- [ ] **Confirm course submission requirements.** **Owner: Saisha.** Check the latest Piazza and
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
      A [189-image candidate](results/df40_transfer/new_data/FRESH_CANDIDATE_REVIEW.md)
      is on CARC. Three pairs were excluded during pre-score quality review;
      provenance limitations and the frozen exploratory protocol are recorded.
      The earlier inference job 14525004 remains held after visual inspection;
      V100 job 14529608 failed the reference check; P100 retry 14742998 passed
      and produced [exploratory results](results/df40_transfer/new_data/fresh_candidate_results_v3/README.md). A separate
      [CPU candidate run](results/df40_transfer/new_data/cpu_candidate_results_v1/README.md)
      completed, but its reference replay exceeded the numerical tolerances;
      CPU and P100 comparisons show one decision difference at 0.5 and nine
      at the CNN baseline validation threshold. Jad's review of the two
      exclusions remains open.
- [ ] **Decide whether to include more manipulation methods.** **Owner: Jad.** The primary
      DF40 tests cover SimSwap, FOMM and Wav2Lip; the exploratory five-method
      subset also has scored InSwapper and PIRender results. Add methods only with
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
      The frozen face-adapted head has now been compared on the 189-image
      exploratory subset; it does not establish person-independent transfer.
      Defer additional training unless a new test and protocol are protected.
- [ ] **Coordinate the frontier-model comparison with its owner.** Include the
      teammate's verified results if ready; that work remains theirs. The
      [dataset handoff](results/df40_transfer/FRONTIER_DATA_HANDOFF.md) describes
      the still-photo pilot and optional five-method candidate, shared-image
      comparison, prompting choices, access and limitations.

## Completed and available

- Resumable OpenAI face runner, frozen protocol/manifest checks, per-image
  checkpoints, concurrent-writer locking, bounded transient retries and
  independent summary/comparison scripts; 17 focused tests passed.
- OpenAI face evaluation: 189/189 successful predictions and 1,134 matched
  model rows, archived locally and pushed on `feature/matched-frontier-comparison`.
  CSV regeneration matched exactly; the report draft records class-specific
  results and dataset limitations. This work is not yet in `main`.
- Cross-branch inventory: Claude's three 630-image GenImage views, the new GPT
  GenImage implementation/results (`7be92ec`) and the extended ViT analysis
  were reviewed for availability. Integration and the remaining audits are
  tracked above; this is not a blanket acceptance of those experiments.
- Corrected clean CNN and ViT benchmarks and the 192-pair comparison.
- Eight-holdout ViT JPEG/resize robustness and the corresponding CNN rerun.
  All 24 affected CNN clean runs matched the original scores on V100; the
  updated 144-row matched comparison passed its replay checks. Earlier P100
  results remain available for provenance.
- Exploratory DF40 face-transfer, FaceForensics++ calibration/adaptation,
  still-photo, paired-error and generator-objective analyses. Their remaining
  data and generalization limitations are recorded in the
  [experiment log](results/df40_transfer/FINAL_REPORT_EXPERIMENT_LOG.md).

**Recommended next step:** integrate the saved branch outputs, resolve the
documented matching/provenance gaps, then regenerate one clear analysis for
final poster/report use. Keep GenImage and face results separate.
New training or a new dataset is optional; the current face set remains exploratory.
