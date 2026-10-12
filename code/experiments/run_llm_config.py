"""Re-run ONE LLM configuration on PrimeVul Paired.

Run one at a time, e.g.:
  python run_llm_config.py --config codellama_zs
  python run_llm_config.py --config deepseek_zs
  python run_llm_config.py --config codellama_fs2
  python run_llm_config.py --config deepseek_fs2
  python run_llm_config.py --config codellama_ft
  python run_llm_config.py --config deepseek_ft

Outputs (in --out_dir):
  preds_<config>.csv   per-sample idx, cve, cwe, target, p_vuln, pred  (reusable for RQ2)
  metrics_<config>.json Acc/Prec/Rec/F1/AUC + PrimeVul pairwise metrics
"""
import argparse, json, math, os, random, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "codellama": "codellama/CodeLlama-7b-hf",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",  # HF name of "DeepSeek-Coder-7B-Instruct"
}
SEED = 42
MAX_CODE_TOKENS = 1500   # per function in the prompt
MAX_FS_EXAMPLE_TOKENS = 600

INSTRUCTION = ("You are a security expert. Does the following C/C++ function contain "
               "a security vulnerability? Answer with YES or NO only.")


# ----------------------------------------------------------------- data
def fn_name(code):
    m = re.search(r"([A-Za-z_][\w:~]*)\s*\(", str(code))
    return m.group(1) if m else None


def build_pairs(df):
    names = df["func"].map(fn_name).tolist()
    t, c = df["target"].tolist(), df["cve"].tolist()
    pairs, i = [], 0
    while i < len(df) - 1:
        if t[i] == 1 and t[i + 1] == 0 and c[i] == c[i + 1] and names[i] == names[i + 1]:
            pairs.append((i, i + 1)); i += 2
        else:
            i += 1
    return pairs


def check_balanced(df, name):
    rate = df.target.mean()
    print(f"[{name}] n={len(df)} vulnerable rate={rate:.3f}")
    if abs(rate - 0.5) > 0.02:
        raise SystemExit(f"[{name}] is NOT balanced ({rate:.1%} vulnerable) -> not PrimeVul Paired. Stop.")


# ----------------------------------------------------------------- prompting
def truncate(tok, code, n):
    ids = tok(code, add_special_tokens=False).input_ids
    return code if len(ids) <= n else tok.decode(ids[:n]) + "\n/* ...truncated... */"


def make_prompt(tok, family, code, shots=None):
    body = INSTRUCTION + "\n\n"
    for ex_code, ex_label in (shots or []):
        body += f"Function:\n```c\n{ex_code}\n```\nAnswer: {'YES' if ex_label else 'NO'}\n\n"
    body += f"Function:\n```c\n{code}\n```\nAnswer:"
    if family == "deepseek":
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True) + "Answer:"
    return body


def answer_ids(tok):
    def ids(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                enc = tok(v, add_special_tokens=False).input_ids
                if enc: out.add(enc[0] if tok.decode(enc[0]).strip() else enc[-1])
        return sorted(out)
    yes, no = ids(["YES", "Yes", "yes"]), ids(["NO", "No", "no"])
    print("YES ids:", [tok.decode(i) for i in yes], "| NO ids:", [tok.decode(i) for i in no])
    assert not set(yes) & set(no)
    return yes, no


@torch.no_grad()
def score(model, tok, prompts, yes, no, bs=1):
    """p(vulnerable) = P(YES-tokens) / (P(YES)+P(NO)) at the first answer position.
    Gives a real ranking score, so AUC is meaningful (no free-text parsing)."""
    probs = []
    for k in range(0, len(prompts), bs):
        enc = tok(prompts[k:k + bs], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits, -1)
        py, pn = p[:, yes].sum(-1), p[:, no].sum(-1)
        probs.extend((py / (py + pn)).cpu().tolist())
        if k % 100 == 0: print(f"  scored {k}/{len(prompts)}", flush=True)
    return np.array(probs)


def pick_shots(train, tok, rng):
    """Two class-balanced shots = one vulnerable/patched pair (short functions only)."""
    pairs = build_pairs(train)
    short = [(a, b) for a, b in pairs
             if len(tok(train.func[a], add_special_tokens=False).input_ids) < MAX_FS_EXAMPLE_TOKENS]
    a, b = rng.choice(short)
    print(f"Few-shot pair: idx {train.idx[a]} (vuln) / {train.idx[b]} (patched), CVE {train.cve[a]}")
    return [(train.func[a], 1), (train.func[b], 0)]


# ----------------------------------------------------------------- fine-tuning (LoRA, paper config)
def finetune(model, tok, family, train, valid, yes, no, args, tag):
    import time
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model.config.use_cache = False
    if args.quant != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_updates = math.ceil(len(train) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_updates), total_updates)  # 10% warmup, linear decay
    print(f"lr={args.lr}  epochs={args.epochs}  grad_accum={args.grad_accum}  optimizer updates={total_updates}")
    yes_t = torch.tensor(yes, device=model.device)
    no_t = torch.tensor(no, device=model.device)
    # build prompts once (same truncation + template as scoring)
    tprompts = [make_prompt(tok, family, truncate(tok, c, MAX_CODE_TOKENS)) for c in train.func]
    vprompts = [make_prompt(tok, family, truncate(tok, c, MAX_CODE_TOKENS)) for c in valid.func]
    tlabels = train.target.values
    best, best_state, bad, log = -1.0, None, 0, []
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); order = rng.permutation(len(train)); opt.zero_grad()
        run_loss, n_seen = 0.0, 0
        for step, i in enumerate(order):
            enc = tok(tprompts[i], return_tensors="pt").to(model.device)
            last = model(**enc).logits[0, -1, :].float()
            # 2-class logits = log-sum over ALL yes/no token variants -> identical to how we score
            two = torch.stack([torch.logsumexp(last[yes_t], 0), torch.logsumexp(last[no_t], 0)]).unsqueeze(0)
            label = torch.tensor([0 if tlabels[i] == 1 else 1], device=model.device)
            loss = torch.nn.functional.cross_entropy(two, label)
            (loss / args.grad_accum).backward()
            run_loss += loss.item(); n_seen += 1
            last_step = step + 1 == len(order)
            if (step + 1) % args.grad_accum == 0 or last_step:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if step % 250 == 0 or last_step:
                el = time.time() - t0
                done = ep * len(order) + step + 1
                eta = el / done * (args.epochs * len(order) - done)
                print(f"  ep{ep} step {step+1}/{len(order)}  avg loss {run_loss/n_seen:.4f}  "
                      f"elapsed {el/60:.1f} min  train ETA {eta/60:.1f} min", flush=True)
        model.eval()
        pv = score(model, tok, vprompts, yes, no)
        val_auc = roc_auc_score(valid.target, pv)
        val_f1 = f1_score(valid.target, pv > 0.5)
        sel = val_auc if args.select == "auc" else val_f1
        print(f"EPOCH {ep}: train loss {run_loss/n_seen:.4f}  val AUC {val_auc:.4f}  val F1 {val_f1:.4f}  "
              f"val pos-rate {(pv > 0.5).mean():.3f}", flush=True)
        log.append(dict(epoch=ep, train_loss=run_loss / n_seen, val_auc=val_auc, val_f1=val_f1,
                        val_pos_rate=float((pv > 0.5).mean())))
        json.dump(log, open(os.path.join(args.out_dir, f"trainlog_{tag}.json"), "w"), indent=2)
        if sel > best:
            best, bad = sel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
            model.save_pretrained(os.path.join(args.out_dir, f"adapter_{tag}_best"))
            print(f"  -> new best (epoch {ep}); adapter saved", flush=True)
        else:
            bad += 1
            if bad >= args.patience: print("  early stopping"); break
    missing = model.load_state_dict(best_state, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    print(f"Loaded best adapter (val {args.select}={best:.4f})")
    return model


# ----------------------------------------------------------------- metrics
def metrics(df, p):
    y, yhat = df.target.values, (p > 0.5).astype(int)
    m = dict(n=len(y), vuln_rate=float(y.mean()), pos_pred_rate=float(yhat.mean()),
             acc=accuracy_score(y, yhat), prec=precision_score(y, yhat, zero_division=0),
             rec=recall_score(y, yhat), f1=f1_score(y, yhat), auc=roc_auc_score(y, p))
    # sanity guard: accuracy can never exceed (benign share) + (vuln share) * recall
    if m["acc"] > (1 - y.mean()) + y.mean() * m["rec"] + 1e-6:
        raise RuntimeError("Accuracy exceeds its ceiling given recall -> evaluation bug.")
    # PrimeVul pairwise metrics
    pairs = build_pairs(df)
    pc = pv = pb = pr = 0
    for a, b in pairs:
        va, vb = yhat[a], yhat[b]
        if va == 1 and vb == 0: pc += 1
        elif va == 1 and vb == 1: pv += 1
        elif va == 0 and vb == 0: pb += 1
        else: pr += 1
    n = max(len(pairs), 1)
    m.update(n_pairs=len(pairs), P_C=pc / n, P_V=pv / n, P_B=pb / n, P_R=pr / n,
             pair_rank_acc=float(np.mean([p[a] > p[b] for a, b in pairs])))  # threshold-free, bias-free
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=[f"{f}_{s}" for f in MODELS for s in ("zs", "fs2", "ft")])
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="llm_results")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--select", choices=["auc", "f1"], default="auc")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none",
                    help="use 4bit on a T4/free Colab; record it in the paper")
    ap.add_argument("--limit", type=int, default=0, help="debug: score only first N test rows")
    ap.add_argument("--train_limit", type=int, default=0, help="smoke test: train on first N rows only")
    ap.add_argument("--valid_limit", type=int, default=0, help="smoke test: validate on first N rows only")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    family, setting = args.config.split("_")
    os.makedirs(args.out_dir, exist_ok=True)

    load = lambda s: pd.read_csv(os.path.join(args.data_dir, f"{s}.csv")).reset_index(drop=True)
    train, valid, test = load("train"), load("valid"), load("test")
    for d, n in ((train, "train"), (valid, "valid"), (test, "test")): check_balanced(d, n)
    if args.limit: test = test.iloc[:args.limit].reset_index(drop=True)
    if args.train_limit: train = train.iloc[:args.train_limit].reset_index(drop=True)
    if args.valid_limit: valid = valid.iloc[:args.valid_limit].reset_index(drop=True)
    smoke = bool(args.limit or args.train_limit or args.valid_limit)
    tag = args.config + ("_SMOKE" if smoke else "")   # smoke runs never overwrite real results
    print(f"run tag={tag}  train={len(train)} valid={len(valid)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(MODELS[family])
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kw = dict(torch_dtype=dtype, device_map="auto")
    if args.quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=dtype)
                                     if args.quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype={dtype}  quant={args.quant}")
    model = AutoModelForCausalLM.from_pretrained(MODELS[family], **kw)
    model.eval()
    yes, no = answer_ids(tok)

    shots = None
    if setting == "fs2":
        shots = pick_shots(train, tok, random.Random(SEED))
        shots = [(truncate(tok, c, MAX_FS_EXAMPLE_TOKENS), l) for c, l in shots]
    if setting == "ft":
        model = finetune(model, tok, family, train, valid, yes, no, args, tag); model.eval()

    prompts = [make_prompt(tok, family, truncate(tok, c, MAX_CODE_TOKENS), shots) for c in test.func]
    p = score(model, tok, prompts, yes, no)

    out = test[["idx", "cve", "cwe", "target"]].copy()
    out["p_vuln"], out["pred"] = p, (p > 0.5).astype(int)
    out.to_csv(os.path.join(args.out_dir, f"preds_{tag}.csv"), index=False)
    m = metrics(test, p); m["config"] = tag
    json.dump(m, open(os.path.join(args.out_dir, f"metrics_{tag}.json"), "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
