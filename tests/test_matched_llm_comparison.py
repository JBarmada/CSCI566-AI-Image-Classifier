import csv
import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "results" / "llm_eval" / "compare_matched.py"
SPEC = importlib.util.spec_from_file_location("compare_matched", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MatchedComparisonTest(unittest.TestCase):
    def test_builds_six_model_matched_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vit_root = root / "vit"
            cnn_root = root / "cnn"
            vit_root.mkdir()
            real = [f"real-{index}" for index in range(10)]
            task = []
            all_hashes = set(real)

            for holdout in ("ADM", "BigGAN", "glide", "Midjourney", "SD14", "SD15", "VQDM", "wukong"):
                fake = [f"{holdout}-fake-{index}" for index in range(10)]
                all_hashes.update(fake)
                pilot = [(sha, 0) for sha in real] + [(sha, 1) for sha in fake]
                task.extend(
                    {
                        "heldout_generator": holdout,
                        "sha256": sha,
                        "path": f"{sha}.jpg",
                        "label": str(label),
                    }
                    for sha, label in pilot
                )
                vit_rows = []
                for mode in MODULE.MODES:
                    vit_rows.extend(
                        {
                            "sha256": sha,
                            "label": str(label),
                            "prob_fake": "0.1" if label == 0 else "0.9",
                            "split": "test",
                            "k": "7",
                            "rep": "0",
                            "seed": "17",
                            "mode": mode,
                            "condition": "clean",
                        }
                        for sha, label in pilot
                    )
                    run = cnn_root / f"holdout_{holdout}_k7_rep0__{mode}__seed17"
                    run.mkdir(parents=True)
                    self.write_csv(
                        run / "test_predictions.csv",
                        [
                            {
                                "sha256": sha,
                                "label": str(label),
                                "probability_fake": "0.1" if label == 0 else "0.9",
                            }
                            for sha, label in pilot
                        ],
                    )
                with gzip.open(vit_root / f"{holdout}.csv.gz", "wt", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(vit_rows[0]))
                    writer.writeheader()
                    writer.writerows(vit_rows)

            scores = {sha: 0.1 if sha.startswith("real-") else 0.9 for sha in all_hashes}
            rows = MODULE.build_rows(task, scores, scores, vit_root, cnn_root)
            self.assertEqual(len(rows), 160)
            self.assertEqual(MODULE.metrics(rows, "gemini")["accuracy"], "1")
            self.assertEqual(MODULE.metrics(rows, "vit_augmented")["auroc"], "1")

    def test_rejects_invalid_scores_and_duplicates(self):
        for score in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                MODULE.valid_score(score)
        with self.assertRaises(ValueError):
            MODULE.index_predictions([{"sha256": "same"}, {"sha256": "same"}])

    def test_excludes_degraded_vit_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vit.csv"
            row = {"sha256": "image", "label": "0", "prob_fake": "0.2",
                   "split": "test", "k": "7", "rep": "0", "seed": "17"}
            rows = []
            for mode in MODULE.MODES:
                rows.extend([{**row, "mode": mode, "condition": "clean"},
                             {**row, "mode": mode, "condition": "jpeg60", "prob_fake": "0.9"}])
            self.write_csv(path, rows)
            scores = MODULE.vit_scores(path, "ADM", {"image": 0})
            self.assertEqual(scores["baseline"]["image"], 0.2)

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
