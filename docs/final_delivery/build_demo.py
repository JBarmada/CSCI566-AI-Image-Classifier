"""Embed verified, anonymized saved scores in the offline demo."""
import json
from pathlib import Path
from build_figures import view_data, face_data, DF
import pandas as pd

OUT = Path(__file__).resolve().parent


def genimage():
    data = view_data().sort_index()
    rows = []
    for key, r in data.iterrows():
        scores = {c: float(r[c])/100 for c in data.columns if c != "label"}
        rows.append({"id": key[:12], "label": int(r.label), "group": "GenImage", "scores": scores})
    return rows


def faces():
    data = face_data()
    rows = []
    for key, g in data.groupby("sha256"):
        assert g.label.nunique() == 1 and g.method.nunique() == 1
        assert ((g.score >= .5).astype(int) == g.prediction).all()
        rows.append({"id": key[:12], "label": int(g.label.iloc[0]), "group": g.method.iloc[0],
                     "scores": {f"{r.model} {r.mode}": float(r.score) for r in g.itertuples()}})
    return rows


def main():
    datasets = {"GenImage: paired presentation": genimage(), "DF40: face crops": faces()}
    assert len(datasets["GenImage: paired presentation"]) == 590
    assert len(datasets["DF40: face crops"]) == 189
    for rows in datasets.values():
        assert len({r["id"] for r in rows}) == len(rows)
        assert all(0 <= v <= 1 for r in rows for v in r["scores"].values())
    template = (OUT/"demo_template.html").read_text(encoding="utf-8-sig")
    assert template.count("/*SAVED_DATA*/") == 1
    summary = pd.read_csv(DF / "clean_cnn_audit/matched_confidence_summary.csv")
    summary = summary[["k", "mode", "cnn_accuracy", "vit_accuracy"]]
    template = template.replace("/*DIVERSITY_DATA*/", summary.to_json(orient="records"))
    (OUT/"demo.html").write_text(template.replace("/*SAVED_DATA*/", json.dumps(datasets)),
                                 encoding="utf-8")
    print("Embedded 590 paired GenImage sources and 189 face crops.")


if __name__ == "__main__":
    main()
