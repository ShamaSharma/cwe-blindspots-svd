"""
verify_selection_metric.py
============================
DIAGNOSTIC ONLY -- not for real model selection.

Trains CNN for the full 10 epochs (no early stopping), and at EVERY epoch
records both validation and test metrics side-by-side. This lets you see
directly, with real numbers, whether epochs with high val_F1 (fixed 0.5
threshold) actually correspond to good test performance, or whether they're
threshold artifacts -- versus whether val_AUC tracks test performance more
reliably.

This peeks at test metrics per-epoch, which you would NEVER do for real
model selection (that's test-set leakage). It's only here to let you verify
the diagnosis with your own eyes before trusting --select_by auc.

Usage:
    python verify_selection_metric.py --model cnn
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from torch.utils.data import Dataset, DataLoader

SEED = 42
VOCAB_SIZE = 10000
EMBED_DIM = 128
HIDDEN = 128


def set_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VulnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.conv = nn.Conv1d(EMBED_DIM, HIDDEN, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x)).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.rnn = nn.LSTM(EMBED_DIM, HIDDEN, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=bidirectional)
        out_dim = HIDDEN * 2 if bidirectional else HIDDEN
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(out_dim, 1)
        self.bidirectional = bidirectional

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.rnn(x)
        x = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(x)).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.rnn = nn.GRU(EMBED_DIM, HIDDEN, num_layers=2, batch_first=True,
                           dropout=0.3, bidirectional=bidirectional)
        out_dim = HIDDEN * 2 if bidirectional else HIDDEN
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(out_dim, 1)
        self.bidirectional = bidirectional

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        x = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(x)).squeeze(-1)


def build_model(name):
    return {
        "cnn":    lambda: CNNModel(),
        "lstm":   lambda: LSTMModel(bidirectional=False),
        "bilstm": lambda: LSTMModel(bidirectional=True),
        "gru":    lambda: GRUModel(bidirectional=False),
        "bigru":  lambda: GRUModel(bidirectional=True),
    }[name]()


def best_threshold_f1(y_true, y_prob):
    """Find the F1-optimal threshold on this set (diagnostic use only --
    shows whether F1 recovers once you stop forcing a fixed 0.5 cut)."""
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_i = np.argmax(f1s[:-1]) if len(thresh) > 0 else 0
    best_t = thresh[best_i] if len(thresh) > 0 else 0.5
    return f1_score(y_true, (y_prob >= best_t).astype(int)), best_t


def get_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            p = torch.sigmoid(model(X_b.to(device))).cpu().numpy()
            probs.extend(p)
            labels.extend(y_b.numpy())
    return np.array(probs), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "lstm", "bilstm", "gru", "bigru"], default="cnn")
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/VulnDetection_Experiment")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"*** MODEL: {args.model.upper()} ***\n")

    FEAT = f"{args.base_dir}/features"
    X_train = np.load(f"{FEAT}/X_train_emb.npy")
    X_valid = np.load(f"{FEAT}/X_valid_emb.npy")
    X_test = np.load(f"{FEAT}/X_test_emb.npy")
    y_train = np.load(f"{FEAT}/y_train.npy")
    y_valid = np.load(f"{FEAT}/y_valid.npy")
    y_test = np.load(f"{FEAT}/y_test.npy")

    train_loader = DataLoader(VulnDataset(X_train, y_train), batch_size=64, shuffle=True)
    valid_loader = DataLoader(VulnDataset(X_valid, y_valid), batch_size=64, shuffle=False)
    test_loader = DataLoader(VulnDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = build_model(args.model).to(device)
    pos_weight_val = (y_train == 0).sum() / y_train.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, dtype=torch.float).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"{'Ep':>3} {'val_AUC':>8} {'val_F1@.5':>10} {'val_F1@opt':>11} "
          f"{'test_AUC':>9} {'test_F1@.5':>11} {'test_F1@opt':>12}")
    print("-" * 70)

    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_prob, val_true = get_probs(model, valid_loader, device)
        test_prob, test_true = get_probs(model, test_loader, device)

        val_auc = roc_auc_score(val_true, val_prob)
        test_auc = roc_auc_score(test_true, test_prob)
        val_f1_fixed = f1_score(val_true, (val_prob >= 0.5).astype(int), zero_division=0)
        test_f1_fixed = f1_score(test_true, (test_prob >= 0.5).astype(int), zero_division=0)
        val_f1_opt, _ = best_threshold_f1(val_true, val_prob)
        test_f1_opt, _ = best_threshold_f1(test_true, test_prob)

        print(f"{epoch:3d} {val_auc:8.4f} {val_f1_fixed:10.4f} {val_f1_opt:11.4f} "
              f"{test_auc:9.4f} {test_f1_fixed:11.4f} {test_f1_opt:12.4f}")

        rows.append({"epoch": epoch, "val_auc": val_auc, "val_f1_fixed": val_f1_fixed,
                     "val_f1_opt": val_f1_opt, "test_auc": test_auc,
                     "test_f1_fixed": test_f1_fixed, "test_f1_opt": test_f1_opt})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_by_auc = df.loc[df["val_auc"].idxmax()]
    best_by_f1 = df.loc[df["val_f1_fixed"].idxmax()]

    print(f"\nIf you select checkpoint by highest val_AUC:")
    print(f"  -> epoch {int(best_by_auc['epoch'])}, "
          f"test_AUC={best_by_auc['test_auc']:.4f}, test_F1@0.5={best_by_auc['test_f1_fixed']:.4f}")

    print(f"\nIf you select checkpoint by highest val_F1 (fixed 0.5 threshold):")
    print(f"  -> epoch {int(best_by_f1['epoch'])}, "
          f"test_AUC={best_by_f1['test_auc']:.4f}, test_F1@0.5={best_by_f1['test_f1_fixed']:.4f}")

    print(f"\nCorrelation between val_AUC and test_AUC across epochs: "
          f"{df['val_auc'].corr(df['test_auc']):.3f}")
    print(f"Correlation between val_F1@0.5 and test_F1@0.5 across epochs: "
          f"{df['val_f1_fixed'].corr(df['test_f1_fixed']):.3f}")
    print("\n(Higher correlation = that metric's epoch-to-epoch validation signal "
          "more reliably predicts what actually happens on test. "
          "If val_F1's correlation is much lower than val_AUC's, that confirms "
          "F1 is the noisier, less trustworthy selection signal here.)")


if __name__ == "__main__":
    main()
