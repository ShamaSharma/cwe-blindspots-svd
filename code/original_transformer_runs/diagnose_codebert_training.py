"""
diagnose_codebert_training.py
===============================
Runs ~100 training steps of CodeBERT fine-tuning while printing gradient
norm and live prediction distribution every 10 steps. This tells us
directly whether the model is actually updating in a meaningful way,
rather than waiting a full 13-minute epoch to find out from an aggregate
loss number.

Usage:
    python diagnose_codebert_training.py
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--accum_steps", type=int, default=1,
                     help="Gradient accumulation steps. 1=no accumulation (batch=32 effective). "
                          "4=effective batch 128, smoother updates.")
args = parser.parse_args()

BASE_OUT = "/content/drive/MyDrive/VulnDetection_Experiment"
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"*** ACTIVE SEED FOR THIS RUN: {SEED} ***")

train_df = pd.read_csv(f"{BASE_OUT}/primevul/train.csv").sample(
    n=500, random_state=SEED).reset_index(drop=True)  # small slice, just for diagnosis

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", num_labels=2).to(device)
model.train()


class CodeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = tokenizer(str(row["func"]), truncation=True, max_length=512,
                         padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["target"]), dtype=torch.long),
        }


loader = DataLoader(CodeDataset(train_df), batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
print(f"*** LR FOR THIS RUN: {args.lr}   ACCUM_STEPS: {args.accum_steps} "
      f"(effective batch = {32 * args.accum_steps}) ***")

print(f"\n{'Step':>5} {'Loss':>8} {'GradNorm':>10} {'PredClass1Rate':>15} {'LogitSpread':>12}")
print("-" * 60)

step = 0
micro_step = 0
optimizer.zero_grad()
for batch in loader:
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    (outputs.loss / args.accum_steps).backward()
    micro_step += 1

    # measure gradient norm BEFORE clipping -- if this is near-zero, gradients
    # aren't flowing at all (dead network / detached graph / frozen params bug).
    # if it's huge (>>1.0), clipping to max_norm=1.0 is crushing the real update.
    total_norm = torch.sqrt(sum(
        p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
    )).item()

    if micro_step % args.accum_steps != 0:
        continue  # keep accumulating, don't step optimizer yet

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        preds = torch.argmax(outputs.logits, dim=-1)
        pred1_rate = preds.float().mean().item()
        logit_spread = (outputs.logits[:, 1] - outputs.logits[:, 0]).abs().mean().item()

    if step % 2 == 0:
        print(f"{step:5d} {outputs.loss.item():8.4f} {total_norm:10.4f} "
              f"{pred1_rate:15.2%} {logit_spread:12.4f}")

    step += 1
    if step >= 20:
        break

print("\n--- Interpretation ---")
print("GradNorm near 0.0 across all steps  -> gradients aren't flowing (real bug: frozen params, detached graph, etc.)")
print("GradNorm large (>>1.0) every step   -> clip_grad_norm_(1.0) may be crushing every real update to a tiny step")
print("LogitSpread staying near 0 the whole time -> classifier head logits for both classes staying nearly tied "
      "(model isn't developing real separation between classes even as loss/preds bounce around)")
