"""Shared loaders for the analysis scripts. All paths are relative to the repository root."""
import ast, glob, os
import numpy as np
import pandas as pd

RQ2_NAMES = {  # file stem -> display name (17 configurations used in the CWE-level analyses)
    "DT": "Decision Tree", "RF": "Random Forest", "LR": "Log. Regression", "SVM": "SVM", "XGB": "XGBoost",
    "CNN": "CNN", "LSTM": "LSTM", "BiLSTM": "BiLSTM", "GRU": "GRU", "BiGRU": "BiGRU",
    "CodeBERT": "CodeBERT", "GraphCodeBERT": "GraphCodeBERT", "UniXcoder": "UniXcoder",
    "CodeLlama-7B_ZS": "CodeLlama (ZS, original run)", "CodeLlama-7B_FS2": "CodeLlama (FS-2, original run)",
    "DeepSeek-Coder-7B-Instruct_ZS": "DeepSeek (ZS, original run)",
    "DeepSeek-Coder-7B-Instruct_FS2": "DeepSeek (FS-2, original run)",
}


def test_index(root):
    t = pd.read_csv(os.path.join(root, "results", "test_index.csv"))
    t["cwe1"] = t["cwe"].map(lambda s: ast.literal_eval(s)[0])
    return t


def pairs(t):
    """Vulnerable row immediately followed by its patched row with the same CVE (431 pairs)."""
    y, c = t["target"].values, t["cve"].values
    return [(t["idx"].iat[i], t["idx"].iat[i + 1], t["cwe1"].iat[i])
            for i in range(len(t) - 1) if y[i] == 1 and y[i + 1] == 0 and c[i] == c[i + 1]]


def rq2_predictions(root):
    """Original-run predictions: {stem: DataFrame indexed by idx with pred_label and (if any) score}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "results", "rq2_predictions", "predictions_*.csv"))):
        stem = os.path.basename(f)[len("predictions_"):-4]
        d = pd.read_csv(f).set_index("idx")
        score = next((c for c in ("probability", "pred_score") if c in d.columns), None)
        d["score"] = d[score] if score else np.nan
        out[stem] = d[["pred_label", "score"]]
    return out


def llm_runs(root):
    """Re-run LLM configurations (probability-scored): {config: DataFrame(idx -> pred, p_vuln)}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "results", "llm", "preds_*.csv"))):
        out[os.path.basename(f)[6:-4]] = pd.read_csv(f).set_index("idx")[["pred", "p_vuln"]]
    return out


def seed_runs(root, model, seed):
    f = os.path.join(root, "results", "transformer_seeds", f"preds_primevul_{model}_seed{seed}.csv")
    return pd.read_csv(f).set_index("idx")[["pred", "p_vuln"]]


def analyzable(t, min_n=10):
    v = t[t.target == 1].set_index("idx")
    n = v["cwe1"].value_counts()
    return v[v["cwe1"].isin(n[n >= min_n].index)], sorted(n[n >= min_n].index)


def degenerate(pred_series, lo=0.05, hi=0.95):
    r = pred_series.mean()
    return r <= lo or r >= hi
