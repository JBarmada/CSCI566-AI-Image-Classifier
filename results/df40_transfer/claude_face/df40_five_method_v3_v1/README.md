# Verified Claude face results

Completed on CARC November 26, 2026 using `claude-opus-5-5`. All 189 images
completed in one attempt each, after a separate two-image smoke. The frozen
face prompt and 50% threshold match the existing OpenAI face study. There was
no retraining, prompt tuning or threshold tuning.

- Accuracy: 70.37%; balanced accuracy: 55.93%; AUROC: 0.6198.
- Detected 121/135 manipulated faces and correctly accepted 12/54 real faces.
- False alarms: 42 real images; missed manipulations: 14 images.
- Fake detections by method: FOMM 27/27, InSwapper 24/27, PIRender 26/27,
  SimSwap 22/27, Wav2Lip 22/27.

Always predicting fake would score 71.43% accuracy and 50% balanced accuracy
on this imbalanced set. Claude's high fake recall accompanies a 77.78% real
false-positive rate. These results do not establish a reliable face detector.
The prior OpenAI run detected 16/135 fakes and flagged 6/54 reals at the same
threshold; the two models make different errors. API preprocessing and output
constraints differ, despite the shared source images and prompt.

Both CARC audits passed. Downloaded prediction checksums, source-code hashes,
response IDs, returned models, stop reasons and frozen manifest membership were
verified. All 189 response-journal entries match every saved prediction field.
Local offline regeneration reproduced all three summary/comparison CSVs byte
for byte, without new API calls. First-to-last saved response span was 354.4
seconds; this excludes setup, initial image validation and the first request's
latency. Full-run usage was 54,999 input and 24,801 output tokens; smoke usage
is separate.

## Files

- [run.json](run.json): frozen protocol, manifest hash and exact selection.
- [predictions.csv](predictions.csv): one score and decision per image.
- [responses.jsonl](responses.jsonl): raw score text, timestamps, response IDs,
  usage and stop reasons.
- [summary/metrics.csv](summary/metrics.csv) and [summary/AUDIT.json](summary/AUDIT.json):
  overall and per-method results, coverage and usage.
- [comparison/metrics.csv](comparison/metrics.csv),
  [matched predictions](comparison/matched_predictions.csv) and
  [comparison audit](comparison/AUDIT.json): Claude plus five saved CNN/ViT
  configurations, exactly 189 sources per configuration (1,134 rows).
- [execution_environment.json](execution_environment.json): actual interpreter,
  SDK version, CARC allocation/node and inference source hashes.
- [LOCAL_REPRODUCTION.json](LOCAL_REPRODUCTION.json): independent local checks.

The five-method set contains 54 real crops and 27 manipulated crops per method,
linked to 27 video-ID pairs. These are not 189 independent identities, and
method/source-role and manipulation-success limitations remain. The face-adapted
ViT is a separate supervised reference, not a zero-shot detector. Per-method
metrics reuse the same 54 real images; do not pool those rows as independent data.

The original outputs remain on CARC at
`/project2/yzhao010_1531/csci566-genimage/results/df40_transfer/claude_face/df40_five_method_v3_v1/`.
The sibling `df40_five_method_v3_v1_smoke/` contains only the operational smoke
and must not be counted as additional evaluation images. See the
[runner guide](../README.md) for reproduction commands.
