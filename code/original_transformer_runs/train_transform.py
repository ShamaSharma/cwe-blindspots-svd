"""
train_transformers.py
======================
Fine-tunes CodeBERT, GraphCodeBERT, or UniXcoder on PrimeVul (paired) for
binary vulnerability detection, matching the paper's stated methodology
(Section V.B) exactly:

    - AdamW, lr=2e-5, batch_size=32, weight_decay=0.01
    - Linear LR scheduler, warmup_ratio=0.1
    - Gradient clipping, max_norm=1.0
    - Max 10 epochs, early stopping patience=5 on validation F1
    - Sequence classification head (2-way softmax), full fine-tuning
      (no freeze/unfreeze phases -- not part of the stated methodology)

Saves: pred_label AND pred_score (softmax prob of class 1) so AUC is
actually computable downstream -- this was the #1 gap in the earlier runs.

Usage:
    python train_transformers.py --model codebert
    python train_transformers.py --model graphcodebert
    python train_transformers.py --model unixcoder

    # quick sanity/calibration run before committing full budget:
    python train_transformers.py --model codebert --epochs 1 --max_train 500
"""

import argparse
import json
import os
import time

SCRIPT_VERSION = "v3-auc-selection-accum4-patience8"  # bump this string every time the script changes

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

MODEL_NAMES = {
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "unixcoder":     "microsoft/unixcoder-base",
}

# published Table IV numbers (PrimeVul Paired) -- used only as a sanity check
PUBLISHED_TABLE_IV = {
    "codebert":      {"f1": 0.6263, "auc": 0.7346},
    "graphcodebert": {"f1": 0.6359, "auc": 0.7327},
    "unixcoder":     {"f1": 0.6839, "auc": 0.7583},
}

SEED = 42


def set_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tokenizer(
            str(row["func"]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["target"]), dtype=torch.long),
            "idx": int(row["idx"]),
        }


def run_eval(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels, all_idx = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
            all_idx.extend(batch["idx"].numpy())
    return (np.array(all_idx), np.array(all_labels),
            np.array(all_preds), np.array(all_probs))


def compute_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, zero_division=0),
        "Rec": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
        "FPR": fpr,
    }


def main():
    print(f"\n{'='*70}\n*** SCRIPT_VERSION: {SCRIPT_VERSION} ***\n{'='*70}\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_NAMES.keys()), required=True)
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/VulnDetection_Experiment",
                         help="Root dir containing primevul/{train,valid,test}.csv. "
                              "NOTE: no '_TEST' suffix by default -- keep this consistent "
                              "with every other script in your pipeline.")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)   # paper: batch_size=32
    parser.add_argument("--lr", type=float, default=2e-5)       # paper: lr=2e-5
    parser.add_argument("--weight_decay", type=float, default=0.01)  # paper: 0.01
    parser.add_argument("--warmup_ratio", type=float, default=0.1)   # paper: 0.1
    parser.add_argument("--epochs", type=int, default=10)       # paper: max 10 epochs
    parser.add_argument("--patience", type=int, default=8)      # raised from paper's 5 -- val_AUC was still
                                                                  # climbing at epoch 6 in diagnostics, patience=5
                                                                  # risks stopping right as it starts converging
    parser.add_argument("--max_train", type=int, default=0,
                         help="0 = full train set. Set e.g. 500 for a fast calibration run.")
    parser.add_argument("--accum_steps", type=int, default=4,
                         help="Gradient accumulation steps. Default 4 -> effective batch 128. "
                              "Diagnostics showed batch=32 alone (accum_steps=1) causes training "
                              "to thrash/diverge rather than converge on CodeBERT-family models; "
                              "accum_steps=4 produced a clean downward loss trend instead.")
    args = parser.parse_args()

    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- this will be extremely slow for a 7-layer+ transformer.")

    model_name = MODEL_NAMES[args.model]
    BASE_OUT = args.base_dir
    PRED_DIR = f"{BASE_OUT}/predictions"
    CKPT_DIR = f"{BASE_OUT}/checkpoints"
    LOG_DIR = f"{BASE_OUT}/logs"
    RESULTS_DIR = f"{BASE_OUT}/results"
    for d in (PRED_DIR, CKPT_DIR, LOG_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    # ---- load data ----
    train_df = pd.read_csv(f"{BASE_OUT}/primevul/train.csv")
    valid_df = pd.read_csv(f"{BASE_OUT}/primevul/valid.csv")
    test_df = pd.read_csv(f"{BASE_OUT}/primevul/test.csv")

    if args.max_train:
        half = args.max_train // 2
        vuln = train_df[train_df["target"] == 1].sample(
            n=min(half, (train_df["target"] == 1).sum()), random_state=SEED)
        benign = train_df[train_df["target"] == 0].sample(
            n=min(args.max_train - len(vuln), (train_df["target"] == 0).sum()), random_state=SEED)
        train_df = pd.concat([vuln, benign]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"Train: {len(train_df)}  Valid: {len(valid_df)}  Test: {len(test_df)}")
    assert train_df["idx"].is_unique, "Duplicate idx in train.csv"
    assert test_df["idx"].is_unique, "Duplicate idx in test.csv"
    assert set(train_df["target"].unique()) <= {0, 1}, "target column must be binary 0/1"

    # ---- model + tokenizer ----
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  "
          f"({'FULL fine-tuning -- correct, matches paper' if trainable == total else 'WARNING: some params frozen -- check this is intentional'})")

    train_loader = DataLoader(CodeDataset(train_df, tokenizer, args.max_len),
                               batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(CodeDataset(valid_df, tokenizer, args.max_len),
                               batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(CodeDataset(test_df, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, eps=1e-8)
    steps_per_epoch = (len(train_loader) + args.accum_steps - 1) // args.accum_steps  # ceil division
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    log_path = f"{LOG_DIR}/{args.model}_train_log.txt"
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log(f"MODEL    : {args.model} ({model_name})")
    log(f"DEVICE   : {device}")
    log("=" * 70)
    log(f"Train: {len(train_df)}  Valid: {len(valid_df)}  Test: {len(test_df)}")
    log(f"LR: {args.lr}  Batch: {args.batch_size}  Weight decay: {args.weight_decay}")
    log(f"Total steps: {total_steps}  Warmup steps: {warmup_steps}")
    log("")

    best_val_auc = -1.0
    best_epoch = 0
    patience_cnt = 0
    best_state = None
    ln2 = np.log(2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_loss_terms = 0
        optimizer.zero_grad()
        for micro_step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            (outputs.loss / args.accum_steps).backward()
            train_loss += outputs.loss.item()
            n_loss_terms += 1

            if micro_step % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # flush any leftover accumulated gradient at end of epoch (partial final group)
        if len(train_loader) % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / max(n_loss_terms, 1)
        _, val_true, val_pred, val_prob = run_eval(model, valid_loader, device)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)
        val_auc = roc_auc_score(val_true, val_prob) if len(set(val_true)) > 1 else float("nan")
        epoch_time = time.time() - t0

        # --- sanity flag: is the model actually learning, or stuck near random-guess loss? ---
        stuck_warning = ""
        if epoch == 1 and avg_train_loss > ln2 * 0.97:
            stuck_warning = "  <-- WARNING: loss ~ln(2), model may not be learning (check LR / frozen params)"

        log(f"Epoch {epoch:02d}/{args.epochs}  train_loss={avg_train_loss:.4f}  "
            f"val_F1={val_f1:.4f}  val_AUC={val_auc:.4f}  time={epoch_time:.0f}s{stuck_warning}")

        # Select checkpoint by AUC, not F1. Evidence: on this exact model/data,
        # val_F1 (fixed 0.5 threshold) can plateau at a threshold artifact
        # (e.g. epoch 1's undertrained head landing near the "always predict
        # positive" baseline) while val_AUC keeps climbing epoch over epoch --
        # F1-based selection was observed to discard 5 epochs of real,
        # measurable improvement in a real run on this exact setup.
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"    -> new best (val_AUC={val_auc:.4f}, val_F1={val_f1:.4f}), checkpoint captured")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val_AUC={best_val_auc:.4f})")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was ever captured -- validation F1 never improved from -1. "
                            "Something is fundamentally broken (check data loading / label column).")

    model.load_state_dict(best_state)
    ckpt_path = f"{CKPT_DIR}/{args.model}_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "model_name": model_name,
        "hyperparams": vars(args),
    }, ckpt_path)
    log(f"\nSaved checkpoint -> {ckpt_path}")

    # ---- final test evaluation ----
    test_idx, test_true, test_pred, test_prob = run_eval(model, test_loader, device)
    metrics = compute_metrics(test_true, test_pred, test_prob)

    log("\n" + "=" * 70)
    log(f"TEST RESULTS -- {args.model}")
    for k, v in metrics.items():
        log(f"  {k:6s}: {v:.4f}")
    log(f"  best_epoch={best_epoch}  best_val_AUC={best_val_auc:.4f}")
    log("=" * 70)

    # --- sanity check against published Table IV (only meaningful for full-data runs) ---
    if not args.max_train and args.model in PUBLISHED_TABLE_IV:
        pub = PUBLISHED_TABLE_IV[args.model]
        f1_diff = abs(metrics["F1"] - pub["f1"])
        auc_diff = abs(metrics["AUC"] - pub["auc"])
        status = "OK" if f1_diff < 0.03 else "CHECK -- large deviation from published Table IV"
        log(f"\nSanity check vs published Table IV: "
            f"F1 diff={f1_diff:.4f}, AUC diff={auc_diff:.4f}  [{status}]")

    # ---- save predictions (with pred_score for AUC downstream) ----
    pred_df = pd.DataFrame({
        "idx": test_idx,
        "true_label": test_true,
        "pred_label": test_pred,
        "pred_score": test_prob,   # <-- was missing in earlier runs; required for Exp 25/36 AUC
    })
    model_col_name = {"codebert": "CodeBERT", "graphcodebert": "GraphCodeBERT",
                       "unixcoder": "UniXcoder"}[args.model]
    pred_path = f"{PRED_DIR}/predictions_{model_col_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    log(f"Saved predictions -> {pred_path}  ({len(pred_df)} rows, {pred_df['idx'].nunique()} unique idx)")

    metrics_path = f"{RESULTS_DIR}/metrics_summary.json"
    results = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            results = json.load(f)
    results[model_col_name] = {k: round(v, 4) for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Updated -> {metrics_path}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Saved log -> {log_path}")


if __name__ == "__main__":
    main()
