"""RQ2 pair-level analysis (Section 4.2.4, Table pairlevel).

Usage: python code/analysis/pair_level.py [--root .]
"""
import argparse, os
import numpy as np
import pandas as pd
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import test_index, pairs, rq2_predictions, degenerate

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); args = ap.parse_args()
t = test_index(args.root)
P = pd.DataFrame(pairs(t), columns=["v", "p", "cwe"])
n = P["cwe"].value_counts(); cats = n[n >= 10].index; P = P[P["cwe"].isin(cats)].reset_index(drop=True)
R = rq2_predictions(args.root)
flag_models = [m for m, d in R.items() if not degenerate(d["pred_label"])]
score_models = [m for m, d in R.items() if d["score"].notna().all() and m != "CNN"]  # CNN: constant predictions
print(f"pairs in analyzable CWEs: {len(P)}; flag-rate models: {len(flag_models)}; score models: {len(score_models)}")

acc = {}
for m in score_models:
    s = R[m]["score"]; dlt = s.reindex(P["v"]).values - s.reindex(P["p"]).values
    acc[m] = np.where(np.abs(dlt) < 1e-9, 0.5, (dlt > 0).astype(float))
A = pd.DataFrame(acc)
rows = {}
for c in cats:
    q = P[P["cwe"] == c]
    fv = np.mean([R[m]["pred_label"].reindex(q["v"]).mean() for m in flag_models])
    fp = np.mean([R[m]["pred_label"].reindex(q["p"]).mean() for m in flag_models])
    rows[c] = {"pairs": len(q), "flagged_vuln": fv, "flagged_patched": fp, "gap": fv - fp,
               "pairwise_acc": A[P["cwe"] == c].mean().mean()}
tab = pd.DataFrame(rows).T.sort_values("flagged_vuln")
print(tab.round(3).to_string())
print(f"overall pairwise accuracy: {A.mean().mean():.3f}")
rng = np.random.default_rng(3); v = A.mean(1).values; lab = P["cwe"].values; obs = tab.loc["CWE-617", "pairwise_acc"]
null = np.array([v[rng.permutation(lab) == "CWE-617"].mean() for _ in range(20000)])
print(f"CWE-617 pairwise accuracy {obs:.3f}; permutation p(harder than chance) = {(null <= obs).mean():.2f}")
os.makedirs(os.path.join(args.root, "results", "derived"), exist_ok=True)
tab.to_csv(os.path.join(args.root, "results", "derived", "table_pairlevel.csv"))
