"""Fine-tune ONE transformer with ONE seed for function-level vulnerability detection.

Matches the paper's Section 3.5 settings:
  AdamW, lr 2e-5, effective batch 32, max 10 epochs, linear schedule with 10% warmup,
  weight decay 0.01, gradient clipping 1.0, early stopping (patience 5) on validation F1.

Data: <data_dir>/train.csv, valid.csv, test.csv with columns  func, target
      (optional: idx, cve  -> enables PrimeVul pair metrics P-C/P-V/P-B/P-R)

Examples
  python run_transformer_seed.py --model codebert --seed 42 --data_dir data/primevul --tag primevul
  python run_transformer_seed.py --model unixcoder --seed 1 --data_dir data/devign  --tag devign

Outputs (in --out_dir):
  preds_<tag>_<model>_seed<seed>.csv     idx, target, p_vuln, pred
  metrics_<tag>_<model>_seed<seed>.json  acc, prec, rec, f1, auc (+ pair metrics), best_epoch
"""
import argparse, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed)

MODELS = {"codebert": "microsoft/codebert-base",
          "graphcodebert": "microsoft/graphcodebert-base",
          "unixcoder": "microsoft/unixcoder-base"}


def encode(tok, df, max_len):
    enc = tok(df["func"].astype(str).tolist(), truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    y = torch.tensor(df["target"].astype(int).values)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


@torch.no_grad()
def predict(model, loader, device, dtype):
    model.eval(); out = []
    for ids, mask, _ in loader:
        with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def pair_metrics(df, yhat):
    """PrimeVul pair metrics: vulnerable row followed by its patched row (same CVE)."""
    if "cve" not in df.columns:
        return {}
    t, c = df["target"].values, df["cve"].values
    pairs = [(i, i + 1) for i in range(len(df) - 1) if t[i] == 1 and t[i + 1] == 0 and c[i] == c[i + 1]]
    if not pairs:
        return {}
    cnt = {"P_C": 0, "P_V": 0, "P_B": 0, "P_R": 0}
    for a, b in pairs:
        if yhat[a] == 1 and yhat[b] == 0: cnt["P_C"] += 1
        elif yhat[a] == 1 and yhat[b] == 1: cnt["P_V"] += 1
        elif yhat[a] == 0 and yhat[b] == 0: cnt["P_B"] += 1
        else: cnt["P_R"] += 1
    return {k: v / len(pairs) for k, v in cnt.items()} | {"n_pairs": len(pairs)}


def scores(y, p):
    yhat = (p > 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat), prec=precision_score(y, yhat, zero_division=0),
                rec=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p), pos_pred_rate=float(yhat.mean())), yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", required=True, help="dataset name used in output file names, e.g. primevul / devign")
    ap.add_argument("--out_dir", default="seed_results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32, help="effective batch size")
    ap.add_argument("--micro_batch", type=int, default=16, help="per-step batch (lower if out of memory)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--select", choices=["f1", "auc"], default="f1")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = f"{args.tag}_{args.model}_seed{args.seed}"
    set_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    load = lambda s: pd.read_csv(os.path.join(args.data_dir, f"{s}.csv")).reset_index(drop=True)
    train, valid, test = load("train"), load("valid"), load("test")
    print(f"[{name}] train={len(train)} valid={len(valid)} test={len(test)}  device={device}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model], num_labels=2).to(device)
    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(encode(tok, train, args.max_len), batch_size=args.micro_batch, shuffle=True, generator=g)
    vl = DataLoader(encode(tok, valid, args.max_len), batch_size=64)
    sl = DataLoader(encode(tok, test, args.max_len), batch_size=64)

    accum = max(1, args.batch // args.micro_batch)
    total = math.ceil(len(tl) / accum) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device == "cuda"))

    best, best_state, best_ep, bad, log = -1.0, None, -1, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); opt.zero_grad(); run = 0.0
        for step, (ids, mask, y) in enumerate(tl):
            with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
            scaler.scale(loss / accum).backward(); run += loss.item()
            if (step + 1) % accum == 0 or step + 1 == len(tl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        pv = predict(model, vl, device, dtype)
        vm, _ = scores(valid["target"].values, pv)
        sel = vm[args.select]
        log.append(dict(epoch=ep, train_loss=run / len(tl), **{f"val_{k}": v for k, v in vm.items()}))
        print(f"[{name}] epoch {ep}: loss {run/len(tl):.4f}  val F1 {vm['f1']:.4f}  val AUC {vm['auc']:.4f}  "
              f"pos-rate {vm['pos_pred_rate']:.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        if sel > best:
            best, best_ep, bad = sel, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{name}] early stopping"); break

    model.load_state_dict(best_state)
    p = predict(model, sl, device, dtype)
    m, yhat = scores(test["target"].values, p)
    m |= pair_metrics(test, yhat)
    m |= dict(model=args.model, dataset=args.tag, seed=args.seed, best_epoch=best_ep,
              best_val=best, select=args.select, minutes=(time.time() - t0) / 60, epochs_log=log)

    out = pd.DataFrame({"idx": test["idx"] if "idx" in test else np.arange(len(test)),
                        "target": test["target"], "p_vuln": p, "pred": yhat})
    out.to_csv(os.path.join(args.out_dir, f"preds_{name}.csv"), index=False)
    json.dump(m, open(os.path.join(args.out_dir, f"metrics_{name}.json"), "w"), indent=2, default=float)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "epochs_log"}, indent=2))


if __name__ == "__main__":
    main()
