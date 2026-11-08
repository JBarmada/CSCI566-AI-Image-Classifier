# Corrected GenImage benchmark

`benchmark` is the accepted image-list release; `vit_benchmark` is the ViT study.

**Completion verified November 8, 2026:** all 192 clean evaluations passed the
final CARC audit and local export checks. See the
[completed analysis handoff](../results/vit_benchmark/README.md) and
[current remaining work](vit-status-2026-11-08.md). Run instructions below are
for reproducibility; the accepted study does not need to be relaunched.
The original `_experiment_manifests`, supplied `_release_v2`, and completed
`results/vit_master_v1` remain intact. Do not mix pilot and benchmark results.

## Corrections and evidence

| Earlier issue | Correction | Acceptance check |
|---|---|---|
| Tests changed across k/repeats | One fixed test set per holdout | Exact identity-set comparison across all 96 experiments |
| Real selections changed | One shared real train/validation/test partition | Identical real identities throughout |
| Supplied real pool had 700/80 train/val | Restore 1400/280/280 per class | Counts checked in all 288 CSVs |
| PNGs versus mixed JPEGs | Select matching Q96-table/4:4:4 real images; reuse converted fakes | Actual tables and sampling checked |
| Plans named old PNGs | Export converted JPEG paths and hashes | Installed bytes and source hashes checked |
| Duplicate or related images | Deduplicate content/identity/pixels; exclude similarity candidates | No exact duplicates or flagged cross-split pairs |

All 16,000 original real-file hashes matched the supplied audit. Of 15,996 distinct
real files, 11,261 match the fake JPEG settings. We use 1,960 real images and 1,960
fake images per generator: 17,640 unique images. Six non-test fake images in three
perceptual-hash candidate pairs were excluded conservatively. This is not a claim
that all near-duplicates are detected.

Training has 1,400 real + 1,400 fake; validation and test each have 280 + 280.
Fake training/validation counts are divided equally among k generators. Reserve
280 fake test images per generator first, then partition the remaining images
into 1,400 training and 280 validation candidates. Original source-folder labels
remain provenance; the new split assignments are authoritative. Image roles do
not change across experiments. Fake test selections from v2 are preserved.
For k=1,2,4, the three repeats use distinct generator combinations. At k=7 there
is only one possible combination; repeats instead vary the sampled fake images.

The release fingerprint and file hashes are in `benchmark/RELEASE.json`.
The installed-data acceptance report is at
`/project2/yzhao010_1531/csci566-genimage/results/benchmark_acceptance.json`.
Small evidence files are in [the benchmark handoff](../results/benchmark/README.md).
Matching JPEG tables/sampling does not match compression history, dimensions or
semantic classes. The real subset is deliberately restricted; claims concern this
benchmark. SD1.4 and SD1.5 count as separate variants, not independent families.

## Run and monitor

The master notebook loads small files in `code/vit/engine/`. Batch execution uses
the same code. Defaults: eight holdouts, k=1/2/4/7, three repeats, seed 17, baseline
and augmented: **192 runs**. Frozen CLIP ViT-L/14 feeds a linear classifier.
Validation AUROC selects checkpoints; differences within 1e-12 are ties resolved
by lower validation BCE. The augmentation recipe and clean testing are unchanged.

From a complete CARC checkout, run full preflight and isolated GPU tests:

```bash
sbatch code/vit/smoke_vit.slurm
```

The smoke job interrupts/resumes both modes on small subsets and verifies exports
and unchanged checkpoints on rerun. Success writes
`results/vit_benchmark_smoke/acceptance.json`. Smoke outputs are not research
results. Repeating the harness requires inspecting existing smoke outputs first;
it refuses to silently skip interruption tests.

After `GPU SMOKE TEST PASSED`:

```bash
sbatch code/vit/run_benchmark.slurm
python code/vit/benchmark_status.py --watch
squeue --me
```

Eight one-GPU jobs run with at most two concurrently. Each holdout has its own
writer lock, cache and results folder. Each requests 24 hours; queue time and
completion time are not guaranteed. Resume one index, for example, with
`sbatch --array=0 code/vit/run_benchmark.slurm`. Index order is ADM, BigGAN, glide,
Midjourney, SD1.4, SD1.5, VQDM, wukong. Do not edit or pull an active checkout.
Verified completed runs are skipped; interrupted runs resume at the last epoch.

Status shows completed/192, current experiment/epoch, batches, last update,
failures and rough active-job remaining estimates. Estimates exclude queue time.
Logs are `vit-benchmark-ARRAY_INDEX.log` in the submission directory.

After obtaining the array ID, schedule the final independent audit:

```bash
sbatch --dependency=afterok:ARRAY_JOB_ID code/vit/finalize_benchmark.slurm
```

Success writes `results/vit_benchmark/summary/ACCEPTANCE.json`, all 192 result rows,
per-holdout/k summaries and equally weighted holdout averages. If training fails,
this dependent job cannot run: fix/resume the affected index and resubmit the final
audit. Repeats reuse test images; they are not independent test-image samples.
CNN must use the same CSVs for matched comparisons; CNN code is unchanged here.

## Rebuild and verify

`code/benchmark/build_benchmark.py` reads the four original Drive ZIP parts and
the JPEG release. Arguments: `--archives`, `--release-zip`, `--inventory` (scan
cache), `--output`. Only reuse an inventory from those exact archives. It refuses
to overwrite a frozen benchmark. Original image bytes are never changed.

`audit_benchmark.py --benchmark PATH --data-root DATA_ROOT --report REPORT.json`
independently checks the release and installed images. Omitting `--data-root`
checks metadata only and cannot satisfy the GPU runner's installed-image gate.
