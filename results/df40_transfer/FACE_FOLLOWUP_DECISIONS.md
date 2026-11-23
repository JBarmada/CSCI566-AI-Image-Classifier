# Face transfer and adaptation: decisions and follow-up

**Owner: Jad. Recommendations prepared for team review; no new dataset was built and no new models were trained for this plan.**

## 1. Fresh face test: recommend yes, but protect the deadline

A fresh test would improve the study because the FaceForensics++ test has
already been examined during threshold selection follow-ups and face-adaptation
experiments. Video IDs were separated under the existing parser, but person
identities, source/target roles and cross-dataset overlap were not established.
A different video name alone is insufficient evidence of a new person. The
existing results remain useful as exploratory findings; a fresh test is not
required to present them honestly.

**Recommendation:** aim for a small, independently checked source-group test
only if the team can trace its origins and verify separation in time. Reserve
it before inspecting scores or choosing methods, prompts, checkpoints or
thresholds. The team should agree on the source and protocol first. If those
checks cannot be met, retain current findings with explicit limitations and
leave a fresh test as future work.

**Pilot plan, not a claimed available dataset:** target at least 30 distinct
traceable real source groups and up to 30 manipulated groups per method,
subject to verified availability. Record one deterministic crop per source
video initially. This is an operational pilot, not a powered sample-size claim.
Use the same images and manifests for all models. Where fake and real come
from the same original, retain the association; split all related source/target
videos together. Require reviewed person IDs where possible and explicitly mark
unknowns. Without evidence for identity separation, call it video-group
separated only. Audit image hashes, labels, method, source, original archive,
source and target IDs where known, frame/crop and processing, split, and reuse
terms. Inspect a documented, fixed sample of crops for visible manipulation;
this does not certify every label.

Use the already fixed original thresholds for a first no-tuning comparison.
If new thresholds are desired, select them on a separate validation partition
and freeze them before the new test. Report false positives and per-method
fake recall together, with counts and uncertainty appropriate to independently
sampled source groups. Do not compare only accuracy when class proportions
differ. Preserve metadata and split manifests, not unlicensed image archives,
in the team repository.

## 2. More methods: recommend a limited extension after provenance checks

The current evaluated DF40 methods are SimSwap, FOMM and Wav2Lip. The official
[DF40 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/34239f60eca7ce9bee5280aaf81362d8-Paper-Datasets_and_Benchmarks_Track.pdf)
and [repository](https://github.com/YZY-stack/DF40) list additional methods
such as InSwapper and BlendFace (swapping) and PIRender (reenactment).
**Recommendation:** if the team has time and authorized access, start with one
additional swap method and one reenactment method, rather than promising all
40. Verify each candidate's actual source, archive, methods, label, crop
procedure, and group overlap before sampling. The fact that DF40 lists a method
does not establish that our team has an accessible, identity-separated subset.
Run only predeclared inference on the frozen shared manifest, and report each
method separately. Do not choose samples based on model scores.

## 3. Adaptation: separate two distinct questions

**Face-adapted linear head:** current training included all three tested
manipulations and the test had already been inspected. Evaluate the frozen,
already-selected head and the existing CNN/ViT baselines on the same fresh,
properly separated face test first. Use the already frozen thresholds, or a
new threshold chosen only on separate validation data. To claim transfer to an
unseen manipulation, exclude that method from all adaptation training and
selection, then evaluate it as a separately planned experiment. This would
require new training and must not use the final test for tuning.

**Generator-objective adapters:** adversarial and contrastive comparisons were
run on frozen CLIP features for eight holdouts and three optimization seeds,
but used one data repeat and previously inspected test sets. Seeds measure
optimization variation, not independent data-split variation. If the team
prioritizes stronger evidence, preregister the comparison, add dataset repeats
or independent splits, hold hyperparameters fixed, and reserve a fresh test.
A [paired review](adaptation_paired_review/README.md) shows the direction
varies across seeds and holdouts; these pairs are not independent. Do not
present the small observed differences as a reliable winner. This is a
separate experiment from face adaptation and may be deferred until after the
poster.

## 4. Limitations for the report and poster

- Released fake labels and hashes do not prove that every crop visibly shows a
  successful manipulation or that a third-party mirror matches official bytes.
- The face datasets differ in source, identity composition, image dimensions,
  video/crop processing, and possible pretraining overlap. These are potential
  confounds, not established explanations for errors.
- Thresholds selected on validation can have different false-positive rates on
  test. The earlier same-data threshold sweep is descriptive, not independent
  calibration evidence.
- The 100 real Celeb-DF crops are reused across manipulation comparisons;
  FaceForensics++ test was inspected before the adaptation follow-up. Do not
  treat these results as independent confirmatory tests.
- A fresh dataset or method extension requires checking access, provenance,
  licenses and team capacity; none is confirmed by this plan.

Detailed evidence: [experiment log](FINAL_REPORT_EXPERIMENT_LOG.md),
[face-adaptation protocol](new_data/ADAPTATION_PROTOCOL.md),
[still-face research](STILL_FACE_RESEARCH.md), and
[three-seed objective results](generator_objectives_seeds/README.md).
