# Separate validation and fresh-test calibration protocol

This tool is prepared but has not fitted any real-data thresholds. Preserve the
existing 500-image dataset and results as prior exploratory evidence. Assemble
and freeze separate validation and fresh-test sets before inspecting results;
do not use the DF40 CDF `val` alias as evidence of independence because the
previous audit found it duplicates `test` paths. Team members should agree on
source rights, manifests, method coverage, and exclusions before release.

Run the four existing frozen configurations on validation using their established
preprocessing. Export one CSV row per image per model and mode, with columns
`sha256,video_group,method,label,model,mode,prob_fake`. `method` must be `real`,
`simswap`, `fomm`, or `wav2lip`; `label` is `0` for real and `1` otherwise.
Every configuration must score the same complete images. Supply the existing
`shared_dataset_v1/manifest.csv` as the previous-test manifest. No model
scores from that set are used to fit the thresholds.

The fixed operating rule is a **single threshold per model across all methods**:
choose the lowest threshold whose validation real-image false-positive rate
is at most 10%. Predictions equal to the threshold count as fake. Ties are
handled by using the next representable floating-point value above an observed
score where necessary. If all real scores equal 1, the threshold can be just
above 1, meaning no image can be flagged. There is no threshold optimization
for fake recall, individual manipulation, or test performance. Save selected
thresholds before any follow-up decisions and never retune on test results.

First fit and write the frozen artifact; no test predictions are read in this
step. Replace the placeholder paths with actual file paths (PowerShell angle
brackets below are placeholders, not literal syntax):

```powershell
cd "C:\Users\creat\OneDrive - University of Southern California\Documents\ChatGPT\CSC566 Proj"
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" deepfake_transfer\calibration_evaluate.py fit --validation VALIDATION.csv --previous-test deepfake_transfer\shared_dataset_v1\manifest.csv --output FROZEN_THRESHOLDS.json
```

Only after preserving that artifact, run the same frozen configurations on the
fresh-test images and export predictions in the same CSV format. Apply the
saved thresholds without refitting:

```powershell
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" deepfake_transfer\calibration_evaluate.py test --frozen FROZEN_THRESHOLDS.json --test FRESH_TEST.csv --previous-test deepfake_transfer\shared_dataset_v1\manifest.csv --output TEST_REPORT.json
```

Both outputs use exclusive creation, so existing artifacts cannot be silently
overwritten. The frozen file retains validation image hashes, groups, and
identity tokens, so the test command can reject overlaps. It also records the
previous-test manifest hash; changing that manifest causes test evaluation to
fail. These artifacts are not cryptographically signed or a substitute for
preserving a dated, team-reviewed experimental record.

The program fails on exact repeated image hashes, video-group names, or known
`idN` identity tokens between the three sets. Those checks are conservative and
limited: renamed or linked source clips, dataset-specific identity meanings,
and numeric YouTube identities cannot be proven independent this way. Resolve
these with verified source/target mapping and report unknown relationships; the
tool reports groups without recognizable identity tokens. The input score
files themselves must be verified against the corresponding image manifests;
the tool cannot independently attest their extraction or model checkpoints.

The output reports validation and fresh-test confusion counts, false-positive
rates, fake recalls, balanced accuracy and descriptive 95% Wilson intervals,
alongside the fresh-test original 0.5 baseline. Intervals assume independent
image observations, which may be optimistic for related identities or sources;
validation intervals do not account for data-driven threshold selection. With
small real samples, the target is only an empirical validation rate, not a
guarantee of future 10% false positives. SimSwap has different sample counts;
balanced accuracy accounts for the class ratio but does not fix source bias.

Synthetic checks can be rerun with:

```powershell
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" -m unittest discover -s deepfake_transfer -p calibration_tests.py -v
```
