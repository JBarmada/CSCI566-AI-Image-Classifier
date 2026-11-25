import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "analysis"))
import common
import errors
import recalibrate


def synthetic_predictions(n=200, runs=(("ADM", 7, 0, "baseline"),), scale=0.4, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for holdout, k, rep, mode in runs:
        for split in ["val", "test"]:
            labels = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
            z = (labels * 2 - 1) * 2.0 + rng.normal(0, 1.5, n)
            frames.append(pd.DataFrame({
                "experiment": f"holdout_{holdout}_k{k}_rep{rep}", "holdout": holdout, "k": k,
                "rep": rep, "seed": 17, "mode": mode, "split": split, "label": labels,
                "image_id": [f"{split}{holdout}{i}" for i in range(n)],
                "path": [f"{holdout}/{split}/{i}.png" for i in range(n)],
                "source_generator": holdout, "prob_fake": common.sigmoid(z * scale)}))
    return common.validate_predictions(pd.concat(frames, ignore_index=True), "synthetic")


class MetricTests(unittest.TestCase):
    def test_ece_hand_computed(self):
        labels = [0, 0, 1, 1]
        probs = [0.05, 0.05, 0.95, 0.55]
        expected = (0.1 + 0.05 + 0.45) / 4
        self.assertAlmostEqual(common.ece(labels, probs), expected)

    def test_perfect_predictions(self):
        m = common.metrics([0, 1], [0.0, 1.0])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["brier"], 0.0)
        self.assertAlmostEqual(m["ece"], 0.0)
        self.assertAlmostEqual(m["overconfidence"], 0.0)

    def test_underconfidence_is_negative(self):
        m = common.metrics([0, 0, 1, 1], [0.4, 0.4, 0.6, 0.6])
        self.assertAlmostEqual(m["overconfidence"], 0.6 - 1.0)

    def test_matches_pipeline_metrics_on_published_data(self):
        path = common.VIT_ROOT / "predictions" / "ADM.csv.gz"
        if not path.exists():
            self.skipTest("published predictions not present")
        frame = pd.read_csv(path)
        recorded = pd.read_csv(common.VIT_ROOT / "summary" / "all_results.csv")
        rec = recorded[recorded.holdout == "ADM"].iloc[0]
        one = frame[(frame.split == "test") & (frame.experiment == rec.experiment)
                    & (frame["mode"] == rec["mode"])]
        self.assertEqual(len(one), rec.n)
        m = common.metrics(one.label, one.prob_fake)
        for col in ["accuracy", "auroc", "ece", "brier", "log_loss", "recall_fake"]:
            self.assertAlmostEqual(m[col], rec[col], places=7)

    def test_summary_averages_repeats_before_holdouts(self):
        runs = pd.DataFrame({"holdout": ["A", "A", "A", "B"], "k": 7, "mode": "baseline",
                             "ece": [0.1, 0.2, 0.3, 0.6]})
        holdout, macro = common.summarize(runs, ["ece"])
        self.assertAlmostEqual(macro.ece_mean.iloc[0], (0.2 + 0.6) / 2)
        self.assertEqual(int(macro.n_holdouts.iloc[0]), 2)


class RecalibrationTests(unittest.TestCase):
    def test_temperature_recovers_known_scale(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 20000)
        z = (y * 2 - 1) * 1.0 + rng.normal(0, 1, len(y))
        calibrated_logit = 2 * z
        probs = common.sigmoid(calibrated_logit * 0.25)
        y = (rng.random(len(y)) < common.sigmoid(calibrated_logit)).astype(int)
        self.assertAlmostEqual(recalibrate.fit_temperature(probs, y), 0.25, delta=0.02)

    def test_threshold_separates_perfectly(self):
        t = recalibrate.fit_threshold([0.1, 0.2, 0.6, 0.7], [0, 0, 1, 1])
        self.assertTrue(0.2 < t < 0.6)

    def test_fit_uses_validation_only(self):
        preds = synthetic_predictions()
        params_a, _, _ = recalibrate.recalibrate(preds)
        changed = preds.copy()
        test = changed.split == "test"
        changed.loc[test, "prob_fake"] = 1 - changed.loc[test, "prob_fake"]
        changed.loc[test, "label"] = 1 - changed.loc[test, "label"]
        params_b, _, _ = recalibrate.recalibrate(changed)
        cols = ["temperature", "platt_a", "platt_b", "val_threshold"]
        pd.testing.assert_frame_equal(params_a[cols], params_b[cols])

    def test_temperature_keeps_decisions_and_fixes_underconfidence(self):
        preds = synthetic_predictions(n=2000)
        _, metrics, test = recalibrate.recalibrate(preds)
        default = metrics[metrics.threshold == "0.5"].set_index("method")
        self.assertEqual(default.loc["raw", "accuracy"], default.loc["temperature", "accuracy"])
        self.assertLess(default.loc["temperature", "ece"], default.loc["raw", "ece"])
        self.assertTrue(((test.prob_raw >= 0.5) == (test.prob_temperature >= 0.5)).all())


class ErrorTests(unittest.TestCase):
    def test_outcomes_and_rates(self):
        frame = pd.DataFrame({
            "experiment": "e", "holdout": "ADM", "k": 7, "rep": 0, "mode": "baseline",
            "label": [1, 1, 0, 0], "prob_raw": [0.4, 0.6, 0.55, 0.3],
            "prob_temperature": [0.05, 0.9, 0.7, 0.1]})
        t = errors.add_outcomes(frame)
        self.assertEqual(list(t.error_type), ["missed_fake", "correct", "false_alarm", "correct"])
        self.assertEqual(list(t.confident_error), [True, False, False, False])
        rates = errors.run_error_rates(t).iloc[0]
        self.assertAlmostEqual(rates.miss_rate, 0.5)
        self.assertAlmostEqual(rates.confident_miss_rate, 0.5)
        self.assertAlmostEqual(rates.false_alarm_rate, 0.5)
        self.assertAlmostEqual(rates.confident_false_alarm_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
