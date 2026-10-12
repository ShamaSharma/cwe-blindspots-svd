"""Check whether a PrimeVul CSV (idx, func, target, cwe, cve) is the Paired split.
Usage: python check_paired.py train.csv valid.csv test.csv
"""
import re, sys
import pandas as pd


def fn_name(code):
    m = re.search(r"([A-Za-z_][\w:~]*)\s*\(", str(code))
    return m.group(1) if m else None


def build_pairs(df):
    """Greedy adjacent pairing: vulnerable row followed by its patched row
    (same CVE, same function name). Returns list of (vuln_pos, patched_pos)."""
    df = df.reset_index(drop=True)
    names = df["func"].map(fn_name)
    pairs, i = [], 0
    while i < len(df) - 1:
        if (df.target[i] == 1 and df.target[i + 1] == 0
                and df.cve[i] == df.cve[i + 1] and names[i] == names[i + 1]):
            pairs.append((i, i + 1)); i += 2
        else:
            i += 1
    return pairs


if __name__ == "__main__":
    for path in sys.argv[1:]:
        df = pd.read_csv(path)
        n, pos = len(df), int(df.target.sum())
        pairs = build_pairs(df)
        print(f"\n== {path}")
        print(f"rows={n}  vulnerable={pos} ({pos/n:.1%})  patched={n-pos}")
        print(f"clean adjacent pairs={len(pairs)}  rows covered={2*len(pairs)/n:.1%}")
        verdict = "PAIRED" if abs(pos / n - 0.5) < 0.02 and 2 * len(pairs) / n > 0.9 else "NOT PAIRED"
        print("verdict:", verdict)
