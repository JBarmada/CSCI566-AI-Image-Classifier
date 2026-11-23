# Audit of the team DF40 construction notebooks

Audited the local untracked notebooks `01_DF40_real_100.ipynb` and
`02_DF40_fake_400.ipynb` against `shared_dataset_v1/manifest.csv`. This is a
static code and manifest audit; the original archives, executed notebook output,
original smoke manifest, and full random selection history were not available
for replay. It does not independently prove that these exact notebook versions
were used to create the uploaded files or that every source label is correct.

## What the notebooks do

Both retrieve DF40 CDF metadata from a Hugging Face mirror via the moving
`main` revision, then reject the metadata unless its SHA-256 equals a hardcoded
value for each method. The real notebook intersects the three methods' listed
real test records and reads an official Celeb-DF-v2 real crop ZIP. The fake
notebook reads SimSwap, FOMM, and Wav2Lip ZIP archives from listed mirrors.
Metadata records and archive paths are matched using a suffix key, not a
verified cryptographic hash for the source archive. The archive member must
be unique for its matching key. This establishes a documented retrieval route,
not independent proof that a third-party mirror is byte-for-byte official.

For each class or method, the notebook sorts candidates and groups, shuffles
with a fresh `random.Random(42)`, shuffles candidate frames within each group,
and takes one readable, unique-byte PNG per group. Unavailable, ambiguous,
unreadable or duplicated candidates are skipped, so exact reproducibility also
depends on archive contents and prior exclusions. It loads the PNG and copies
the original bytes, without a further crop, resizing, recompression, or other
image transformation. The PNG dimensions and SHA-256, metadata path and archive
member are recorded. Upstream face detection, alignment, crop generation, and
compression within the source archives remain unverified, and could differ
between real and fake sources. The inference scripts apply their own existing
model-specific resizing and normalization after dataset construction.

The notebooks require a 24-row prior smoke manifest with six per group and
exclude its exact group strings and hashes. `EXTRA_EXCLUDED_GROUPS` defaults
to an empty set; it can hold known related clip IDs, but neither notebook
automatically traces source/target clip or identity relationships. The fake
notebook also checks the 100 real images against their recorded hashes and
excludes their exact hashes from fake selection. Each fake method is sampled
independently; a group can occur across methods. The input smoke manifest and
actual notebook settings used by the team are not present in this audit, so
the exclusion actually applied cannot be verified from these notebooks alone.

## Shared-manifest checks

- The 500 rows contain 100 real, 200 SimSwap (two batches of 100), 100 FOMM,
  and 100 Wav2Lip; labels and batch names agree with the notebook defaults.
- All 500 metadata paths are distinct; every `video_group` matches its
  metadata path's group folder, and every archive-member path has the same
  group/frame suffix as the corresponding metadata path. These string checks
  do not verify archive bytes.
- Each method has one listed image per group, and the two SimSwap batches have
  200 distinct groups. SimSwap and FOMM share 36 exact group names; the other
  fake-method pairs share no exact group name. Cross-method overlap is expected
  when techniques use related sources and must not be treated as independent.
- The manifest has 62 Celeb-real and 38 YouTube-real images. Parsed `idN` tokens
  occur in 42 real, 53 SimSwap, 43 FOMM and 36 Wav2Lip distinct identity-token
  sets. Of the real tokens, 40 also occur in SimSwap group names, 31 in FOMM,
  and 25 in Wav2Lip. The semantic role of each token is not established here;
  YouTube numeric identifiers are not mapped to people. Treat these as possible
  relationships, not proven matching original/fake pairs.
- An earlier verification independently checked all downloaded file hashes and
  dimensions against this manifest. Agreement with the manifest does not verify
  the provenance or successful manipulation of each crop.

## Implications for separate validation and final test

Do not reuse the notebook unchanged for a new split: its default exclusions
cover only the smoke manifest, not the complete 500-image current set, and
renamed or cross-method groups may share videos or people. Build and record
explicit exclusions across current data, validation and fresh test, with all
known source and target IDs, paths and hashes; keep unresolved identities and
cross-video mappings marked unknown. The same official CDF `val` and `test`
metadata paths were previously observed to be identical, so their names do not
establish independent data. If mappings are incomplete, report a video-group
disjoint evaluation with the identity leakage limitation instead of claiming
identity-disjoint generalization. The current test was inspected and is an
exploratory baseline, not a fresh final test after model-selection decisions.
