"""Historical full GenImage runner; shares the original pilot implementation."""

import run_pilot


if __name__ == "__main__":
    run_pilot.MANIFEST = run_pilot.OUT / "unique_api_manifest.csv"
    for name, settings in run_pilot.MODELS.items():
        settings["output"] = run_pilot.OUT / f"{name}_full_predictions.csv"
    run_pilot.main()
