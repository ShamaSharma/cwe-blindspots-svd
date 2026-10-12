"""RQ2: bias-corrected relative deficit per CWE (Table 5) and the CWE-617 statistics of Section 4.2.2.

Usage: python code/analysis/cwe_deficit.py [--root .] [--perms 20000]
"""
import argparse, os
import numpy as np
import pandas as pd
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import test_index, rq2_predictions, llm_runs, seed_runs, analyzable, degenerate

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--perms", type=int, default=20000)
args = ap.parse_args()
rng = np.random.default_rng(42)
t = test_index(args.root)
va, cats = analyzable(t)
lab = va["cwe1"].values
TARGET = "CWE-617"


def deficit_matrix(preds):
    """preds: {name: Series idx->0/1}; drops degenerate models; returns (names, D models x samples)."""
    keep = [m for m, p in preds.items() if not degenerate(p)]
    D = np.array([preds[m].reindex(va.index).values for m in keep], dtype=float)
    assert not np.isnan(D).any(), "missing predictions for some analyzable samples"
    return keep, D


def deficits(D, labels):
    base = D.mean(1)
    return {c: base - D[:, labels == c].mean(1) for c in cats}


def perm_tests(D, n_perm, by_cve=False):
    obs = deficits(D, lab)[TARGET].mean()
    single, mx = [], []
    if by_cve:
        g = va.groupby("cve")["cwe1"].first(); groups = va["cve"].values
    for _ in range(n_perm):
        if by_cve:
            m = dict(zip(g.index, rng.permutation(g.values))); p = np.array([m[x] for x in groups])
        else:
            p = rng.permutation(lab)
        d = deficits(D, p)
        means = {c: d[c].mean() for c in cats if (p == c).any()}
        single.append(means.get(TARGET, np.nan)); mx.append(max(means.values()))
    single, mx = np.array(single), np.array(mx)
    return obs, np.nanmean(single >= obs), np.mean(mx >= obs)


# ---------------------------------------------------------------- main analysis (original runs)
orig = {k: v["pred_label"] for k, v in rq2_predictions(args.root).items()}
names, D = deficit_matrix(orig)
print(f"Models kept: {len(names)} (excluded as degenerate: {sorted(set(orig) - set(names))})")
d = deficits(D, lab)
tab = pd.DataFrame({c: {"n": int((lab == c).sum()), "mean_relative_deficit": d[c].mean(),
                        "models_deficit_ge_0.15": int((d[c] >= 0.15).sum())} for c in cats}).T
tab = tab.sort_values(["models_deficit_ge_0.15", "mean_relative_deficit"], ascending=False)
print("\nTable 5 (relative deficit):"); print(tab.round(3).to_string())
os.makedirs(os.path.join(args.root, "results", "derived"), exist_ok=True)
tab.to_csv(os.path.join(args.root, "results", "derived", "table5_relative_deficit.csv"))
pd.Series(d[TARGET], index=names).sort_values().to_csv(os.path.join(args.root, "results", "derived", "cwe617_deficit_per_model.csv"))

obs, p1, pmax = perm_tests(D, args.perms)
_, _, pmax_cve = perm_tests(D, args.perms, by_cve=True)
s617 = va[va["cwe1"] == TARGET]; cves = s617["cve"].unique(); pos = {k: i for i, k in enumerate(va.index)}
base = D.mean(1); boot = []
for _ in range(5000):
    cols = [pos[i] for c in rng.choice(cves, len(cves)) for i in s617.index[s617["cve"] == c]]
    boot.append((base - D[:, cols].mean(1)).mean())
tf = s617["cve"].str.startswith("CVE-2022-235")
def sub(mask): return (base - D[:, [pos[i] for i in s617.index[mask]]].mean(1)).mean()
print(f"\n{TARGET}: deficit {obs:.3f}; per-category permutation p = {p1:.4f}; max-statistic p = {pmax:.4f}; "
      f"CVE-level max-statistic p = {pmax_cve:.4f}")
print(f"CVE-clustered bootstrap 95% CI: [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]")
print(f"{len(s617)} samples from {s617['cve'].nunique()} CVEs; TensorFlow ({tf.sum()}): {sub(tf):.3f}; "
      f"other projects ({(~tf).sum()}): {sub(~tf):.3f}")

# ---------------------------------------------------------------- robustness: Table 3 runs
ml_dl = {k: orig[k] for k in ["DT", "RF", "LR", "SVM", "XGB", "CNN", "LSTM", "BiLSTM", "GRU", "BiGRU"]}
llm = {k: v["pred"] for k, v in llm_runs(args.root).items()}
print("\nRobustness (Table 3 runs: re-scored LLMs incl. FT; one transformer seed at a time):")
for s in (42, 1, 2):
    preds = dict(ml_dl, **llm, **{f"{m}_s{s}": seed_runs(args.root, m, s)["pred"] for m in ["codebert", "graphcodebert", "unixcoder"]})
    n2, D2 = deficit_matrix(preds)
    d2 = deficits(D2, lab); rank = sorted(cats, key=lambda c: -d2[c].mean()).index(TARGET) + 1
    o2, q1, qmax = perm_tests(D2, args.perms)
    print(f"  seed {s:>2}: {len(n2)} models, CWE-617 deficit {o2:.3f}, rank {rank}, per-category p = {q1:.4f}, max-statistic p = {qmax:.4f}")
