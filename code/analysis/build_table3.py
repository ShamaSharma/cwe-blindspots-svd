"""RQ1: PrimeVul Paired results (Table 3) incl. P-C, transformer mean +- std over seeds,
and the LLM pairwise binomial tests quoted in Section 4.1.

Usage: python code/analysis/build_table3.py [--root .]
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import test_index, pairs, rq2_predictions, llm_runs, seed_runs

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); args = ap.parse_args()
t = test_index(args.root); P = pairs(t); y = t.set_index("idx")["target"]


def metrics(pred, score):
    yy = y.values; yh = pred.reindex(y.index).values.astype(int); s = score.reindex(y.index).values
    pd_ = dict(zip(y.index, yh))
    return dict(acc=accuracy_score(yy, yh), prec=precision_score(yy, yh, zero_division=0),
                rec=recall_score(yy, yh, zero_division=0), f1=f1_score(yy, yh, zero_division=0),
                auc=roc_auc_score(yy, s), P_C=np.mean([pd_[a] == 1 and pd_[b] == 0 for a, b, _ in P]),
                flagged=yh.mean())


rows = []
R = rq2_predictions(args.root)
for fam, keys in [("ML", ["DT", "RF", "LR", "SVM", "XGB"]), ("DL", ["CNN", "LSTM", "BiLSTM", "GRU", "BiGRU"])]:
    for k in keys:
        rows.append(dict(family=fam, model=k, **metrics(R[k]["pred_label"], R[k]["score"])))
for m in ["codebert", "graphcodebert", "unixcoder"]:
    runs = pd.DataFrame([metrics(seed_runs(args.root, m, s)["pred"], seed_runs(args.root, m, s)["p_vuln"]) for s in (42, 1, 2)])
    rows.append(dict(family="Transformer", model=m, **runs.mean().to_dict(),
                     **{f"{k}_std": v for k, v in runs.std(ddof=1).to_dict().items()}))
L = llm_runs(args.root)
for k in ["codellama_zs", "deepseek_zs", "codellama_ft", "deepseek_ft", "codellama_fs2", "deepseek_fs2"]:
    rows.append(dict(family="LLM", model=k, **metrics(L[k]["pred"], L[k]["p_vuln"])))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T[["family", "model", "acc", "prec", "rec", "f1", "auc", "P_C", "flagged"]].round(4).to_string(index=False))
os.makedirs(os.path.join(args.root, "results", "derived"), exist_ok=True)
T.to_csv(os.path.join(args.root, "results", "derived", "table3_primevul.csv"), index=False)

print("\nLLM pairwise test (vulnerable scored above its patch, ties excluded):")
for k, d in L.items():
    s = d["p_vuln"]; dlt = np.array([s[a] - s[b] for a, b, _ in P]); nt = dlt[np.abs(dlt) > 1e-6]; v = int((nt > 0).sum())
    print(f"  {k:14s} {v}/{len(nt)}  binomial p = {binomtest(v, len(nt)).pvalue:.4f}")
