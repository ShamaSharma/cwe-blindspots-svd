"""Attention figure for Section 4.3.3 (data: exp34_cwe617_attention.csv)."""
import re, sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]

def parse(s):
    # entries look like  Ġ((0.015)  or  ())(0.010)  -> token text + weight
    return [(m.group(1), float(m.group(2))) for m in re.finditer(r"\s*(.+?)\((\d\.\d+)\)(?:,|$)", s)]

def clean(t):
    t = t.replace("Ġ", "").replace("Ċ", "\\n").strip()
    return t if t else "␣"

def is_delim(t):
    return re.fullmatch(r"[\W_]+", t) is not None   # only punctuation / operators

# summary over all 12 samples (printed for the text)
rows = [(r.idx, r.correct, clean(t), w) for r in d.itertuples() for t, w in parse(r.top_tokens)]
A = pd.DataFrame(rows, columns=["idx", "correct", "tok", "w"])
A["delim"] = A.tok.map(is_delim)
print("samples", A.idx.nunique(), "tokens", len(A), "delimiter share %.1f%%" % (100 * A.delim.mean()))
print("non-delimiter tokens:", sorted(set(A.tok[~A.delim])))
print("top weight range %.3f-%.3f" % (A.groupby('idx').w.max().min(), A.groupby('idx').w.max().max()))

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=300)
for ax, (idx, title) in zip(axes, [(195389, "Detected (idx 195389)"), (194998, "Missed (idx 194998)")]):
    toks = parse(d.loc[d.idx == idx, "top_tokens"].iloc[0])
    w_tok = max(len(clean(t)) for t, _ in toks)
    labels = [f"{i+1:>2}. {clean(t):<{w_tok}}" for i, (t, _) in enumerate(toks)]
    w = [x for _, x in toks]
    colors = ["#4a78b5" if not is_delim(clean(t)) else "#9db5d6" for t, _ in toks]
    ax.barh(range(len(w))[::-1], w, color=colors, height=0.7, zorder=2)
    ax.set_yticks(range(len(w))[::-1]); ax.set_yticklabels(labels, family="monospace", fontsize=8)
    ax.axvline(1 / 512, ls=":", lw=0.8, color="#666666", zorder=3)
    ax.set_title(title, fontsize=9); ax.set_xlabel("Attention weight from [CLS]")
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0); ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(color="#9db5d6", label="Delimiter / operator"), Patch(color="#4a78b5", label="Word token"),
                    Line2D([], [], ls=":", lw=0.8, color="#666666", label="Uniform attention (1/512)")],
           loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=300); print("saved", out)
