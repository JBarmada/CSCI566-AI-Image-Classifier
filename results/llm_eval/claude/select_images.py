import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LLM_EVAL = HERE.parent
SEED = 566


def select(manifest: pd.DataFrame, pilot: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    manifest = manifest.drop_duplicates("sha256").sort_values("sha256").reset_index(drop=True)
    pilot_ids = set(pilot.sha256)
    missing = pilot_ids - set(manifest.sha256)
    if missing:
        raise ValueError(f"{len(missing)} pilot images are not in the manifest")

    manifest["group"] = manifest.source_generator.where(manifest.label == 1, "real")
    picked = []
    for group, pool in manifest.groupby("group", sort=True):
        from_pilot = pool[pool.sha256.isin(pilot_ids)]
        rest = pool[~pool.sha256.isin(pilot_ids)]
        need = per_group - len(from_pilot)
        if need < 0:
            raise ValueError(f"{group}: pilot already has {len(from_pilot)} > {per_group} images")
        if need > len(rest):
            raise ValueError(f"{group}: only {len(rest)} extra images available, need {need}")
        extra = rest.sample(n=need, random_state=seed)
        picked.append(pd.concat([from_pilot.assign(in_pilot=True), extra.assign(in_pilot=False)]))

    out = pd.concat(picked).sort_values(["group", "sha256"]).reset_index(drop=True)
    cols = ["sha256", "path", "label", "source_generator", "group", "in_pilot", "source_split"]
    return out[cols]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pick the shared GenImage image list for zero-shot testing.")
    parser.add_argument("--manifest", type=Path, default=LLM_EVAL / "unique_api_manifest.csv")
    parser.add_argument("--pilot", type=Path, default=LLM_EVAL / "pilot_unique_api_manifest.csv")
    parser.add_argument("--per-group", type=int, default=70)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=HERE / "shared_image_list.csv")
    args = parser.parse_args(argv)

    out = select(pd.read_csv(args.manifest), pd.read_csv(args.pilot), args.per_group, args.seed)
    out.to_csv(args.output, index=False)
    out.path.to_csv(args.output.with_name("carc_paths.txt"), index=False, header=False)

    print(f"Selected {len(out)} images ({out.in_pilot.sum()} from the pilot) -> {args.output}")
    print(out.groupby("group").agg(images=("sha256", "size"), from_pilot=("in_pilot", "sum")).to_string())


if __name__ == "__main__":
    main()
