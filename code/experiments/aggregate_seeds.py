"""Combine per-seed results into mean ± std tables.

Usage:  python aggregate_seeds.py seed_results/
Reads   metrics_<dataset>_<model>_seed<N>.json  (written by run_transformer_seed.py)
Writes  seed_summary.csv  and  seed_summary.tex  (LaTeX rows: mean $\\pm$ std)
"""
import glob, json, os, sys
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "seed_results"
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "metrics_*_seed*.json")))]
if not rows:
    sys.exit(f"no metrics_*_seed*.json files in {d}")
df = pd.DataFrame(rows)
cols = [c for c in ["acc", "prec", "rec", "f1", "auc", "P_C"] if c in df.columns]

print("Runs found:")
print(df.groupby(["dataset", "model"]).seed.apply(lambda s: sorted(s.tolist())).to_string(), "\n")

g = df.groupby(["dataset", "model"])[cols]
mean, std, n = g.mean(), g.std(ddof=1), g.size()
summary = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
summary["n_seeds"] = n
print(summary.to_string())
for (ds, mdl), k in n.items():
    if k < 3:
        print(f"WARNING: {ds}/{mdl} has only {k} seed(s); std needs >= 2, report >= 3.")

out = pd.concat({"mean": mean, "std": std}, axis=1)
out["n_seeds"] = n
out.to_csv("seed_summary.csv")

pretty = {"codebert": "CodeBERT", "graphcodebert": "GraphCodeBERT", "unixcoder": "UniXcoder"}
with open("seed_summary.tex", "w") as fh:
    for ds in mean.index.get_level_values(0).unique():
        fh.write(f"% {ds}  (columns: {' & '.join(cols)})\n")
        for mdl in mean.loc[ds].index:
            cells = [f"{mean.loc[(ds, mdl), c]:.4f} $\\pm$ {std.loc[(ds, mdl), c]:.4f}" for c in cols]
            fh.write(f"{pretty.get(mdl, mdl)} & " + " & ".join(cells) + " \\\\\n")
print("\nwrote seed_summary.csv and seed_summary.tex")
