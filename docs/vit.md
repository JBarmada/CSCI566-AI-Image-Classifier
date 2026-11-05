# ViT workflow

Use the [benchmark guide](benchmark.md) for the corrected 192-run study, GPU smoke
test, batch submission, resume, progress and final verification.

- [Master notebook](../ViT_Master_Experiments.ipynb): interactive entry point.
- [Settings](../code/vit/engine/settings.py): model, data and training options.
- [Runner](../code/vit/run_vit_master.py): executes the same notebook implementation.
- [Slurm launcher](../code/vit/run_benchmark.slurm): eight holdouts, two jobs at once.
- [CARC setup](carc.md): environment and shared storage.
- [Completed ADM pilot](../results/vit_master_v1/README.md): frozen historical results.

Select **CSCI566 ViT** in GPU-backed Jupyter and open the notebook from a complete
checkout. Its code now lives in small files under `code/vit/engine/`. Interactive
defaults run all 192 configurations in `vit_benchmark/notebook`; use batch jobs
for the coordinated study. Never duplicate training alongside those jobs.

The old ADM study used earlier code, retained in Git history. Current code has a
new protocol and must not write into `vit_master_v1`. Transient CLIP function
addresses are still normalized; genuine data, setting, code or environment
changes reject resume. Do not change an active checkout.

The runner takes an OS-managed study lock. Per-run locks are archived only when
the owner can be proved inactive. Ambiguous locks stop execution. Time-limit
interruptions resume from the last full epoch.

CNN crop/flip and ViT blur/JPEG/resize/noise/filtering are different augmentation
treatments. The frozen-feature ViT pipeline is not full backbone fine-tuning,
domain-adversarial learning or zero-shot classification.
