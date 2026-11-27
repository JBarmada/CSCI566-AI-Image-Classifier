"""Check the numbers and source cohorts used by the final review package."""
from pathlib import Path
import numpy as np
import pandas as pd
from build_figures import DF, FR, view_data, face_data


def check_clean():
    data = pd.read_csv(DF/"clean_cnn_audit/matched_confidence_summary.csv")
    base = data[data["mode"] == "baseline"].sort_values("k")
    np.testing.assert_allclose(base.cnn_accuracy*100, [62.75,70.25,80.86,83.30], atol=.005)
    np.testing.assert_allclose(base.vit_accuracy*100, [73.00,77.29,85.42,88.30], atol=.005)
    robust = pd.read_csv(DF/"cnn_robustness_v100/paired_macro.csv")
    resize = robust[robust.condition == "resize"].set_index("mode")
    np.testing.assert_allclose(resize.loc["baseline",["accuracy_cnn","accuracy_vit"]].to_numpy(dtype=float)*100,
                               [50.20,56.96], atol=.005)
    np.testing.assert_allclose(resize.loc["augmented",["accuracy_cnn","accuracy_vit"]].to_numpy(dtype=float)*100,
                               [68.85,81.85], atol=.005)


def check_frontier():
    data = pd.read_csv(FR/"team_readiness_2026-11-26/matched_by_holdout.csv")
    g = data[data.study == "630 source detector224"].groupby("model")
    assert len(g)==6
    means = g.balanced.mean()*100
    for model, expected in [("ViT_baseline",89.37),("CNN_baseline",85.58),
                            ("Claude_detector224",77.79),("GPT_detector224",63.40)]:
        assert abs(means[model]-expected)<.006
    views = view_data()
    fake = views[views.label == 1]
    for name, expected in [("GPT",[284,281,142]),("Claude",[388,388,291])]:
        counts = [(fake[f"{name} {v}"]>=50).sum() for v in ["original","square","detector224"]]
        assert counts==expected


def check_face():
    rows = face_data()
    expected = {("cnn","baseline"):(53,133), ("cnn","augmented"):(31,92),
                ("vit","baseline"):(42,106),("vit","augmented"):(39,96),
                ("vit","face_adapted"):(20,98),("openai","zero_shot"):(6,16),
                ("claude","zero_shot"):(42,121)}
    sets=[]
    for key,g in rows.groupby(["model","mode"]):
        assert ((g.score>=.5).astype(int)==g.prediction).all()
        assert (g[g.label==0].prediction.sum(),g[g.label==1].prediction.sum())==expected[key]
        sets.append(set(zip(g.sha256,g.label)))
    assert len(sets)==7 and all(s==sets[0] for s in sets)


def main():
    check_clean()
    check_frontier()
    check_face()
    print("PASS: clean diversity, robustness, matched frontier, all six view counts, all seven face configurations.")


if __name__=="__main__":
    main()
