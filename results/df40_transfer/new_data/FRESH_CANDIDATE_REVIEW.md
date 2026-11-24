# Five-method face candidate: score-blind data review

**Status: candidate for team review, not a validated independent test. No
model scores have been computed on these images.**

## Selection and contents

On November 23, a deterministic, score-blind procedure selected 30 disjoint
FaceForensics++ video-ID pairs present in metadata for SimSwap, FOMM,
InSwapper and PIRender. It selected a Wav2Lip group referencing one recorded
video ID in each pair. Both real videos and one crop per fake method were
included per pair. It excluded the 484 recorded IDs in the existing FF
test and face-adaptation plans. Group names and selected middle frames are in
the [original plan](fresh_candidate_plan_v1.json); the
[selection script](prepare_candidate_pairs.py) documents ranking and parsing.
The [path audit](fresh_candidate_member_audit.json) checked all 210 paths
against ZIP indexes before fetching image bytes.

One selected pair, `628_568`, had **identical bytes** for the selected SimSwap
and InSwapper image. Four additional matching frame names from that pair were
also byte-identical across those two archives. See the
[duplicate audit](duplicate_pair_audit.json). This does not tell us which
label, if either, is incorrect. The entire seven-image pair was excluded
without reading model scores; it was not replaced.

The retained [review manifest](fresh_candidate_review_v2/manifest.csv) has
**29 pairs: 58 real images and 29 fake images per method, 203 images total**.
All are unique by SHA-256 and none match SHA-256 hashes found in earlier local
project manifests. Image bytes were checked as readable 256×256 PNGs and the
ZIP reads verified CRC. Exact hashes do not detect different frames of the same
video, re-encodings or the same person. Nor does this small check establish
that the other images are correctly labeled.

Jad reviewed a fixed set of the first three retained pairs and reported that
the shown crops looked usable. This checks 21 images visually; it does not
certify successful manipulation, all labels, or the other 182 images. Real and
fake crops from a pair may show different moments; Wav2Lip's driving audio
source is unknown.

The team-accessible image copy is on CARC at
`/project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v2_20261123/`.
All 203 transferred image hashes were checked there. The archive
`fresh_candidate_review_v2.tar.gz` has SHA-256
`05c9ac1d81148c71739eb414ccd44610f9892d28a73a13c9e7b515e71c5e6b4c`.
Images are not committed to GitHub.

## Boundaries and next steps

- This is another subset of the upstream FF training split, not a new external
  dataset. The known video IDs differ from prior project plans, but
  person-level, renamed-video and cross-dataset independence are unverified.
- The fake archives are third-party mirrors. Matching DF40 metadata paths and
  CRCs does not prove bytes match the official release. The duplicate pair
  raises a concrete concern about method labels; more provenance checks are
  needed before claiming that all five manipulations are distinct.
- The [DF40 authors](https://github.com/YZY-stack/DF40#license) state a
  CC BY-NC 4.0 license for the dataset and code. Check original FF++ terms and
  mirror provenance before any wider redistribution; CARC is the restricted
  project area, and the GitHub repository contains manifests, not images.
- Freeze the model checkpoints, preprocessing, metrics and thresholds in a
  documented protocol before any test scoring. Compare existing baselines and
  the already trained face-adapted head on the same retained images. Report
  real false positives and fake recall per method as counts, plus AUROC;
  account for the common real set and paired video groups. Any result is an
  exploratory source-ID-separated follow-up, not a person-independent test.
- Review additional images and investigate method-label provenance before
  treating the extra methods as validated. The final poster and interpretation
  remain subject to Jad and team review.

## Reproducing the selection

The two extra FF metadata files are obtained from the pinned metadata source
used by [audit_extra_methods.py](audit_extra_methods.py): fetch `inswap_ff.json`
and `pirender_ff.json` into `tmp/new_method_metadata/`. Their SHA-256 values,
along with the existing method metadata hashes, are stored in the plan and
[inventory](shared_pair_inventory.json). The five scripts for selection,
index checks, retrieval, duplicate quarantine, and packaging retain the
original plan and audit trail. Image files require the local FF real archive
and access to the pinned public mirrors described in those scripts.
