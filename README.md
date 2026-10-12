# Why Do Vulnerability Detectors Fail? — Replication Package

Code and results for the paper **"Why Do Vulnerability Detectors Fail? A Cross-Family Empirical Analysis of CWE-Level Detection Blind Spots"** (Shama, Sarika Jain).

The package covers the PrimeVul Paired experiments: the LLM and transformer runs behind Table 3, all CWE-level analyses, and the scripts for Figures 6 and 8. Code for the classical ML and DL baselines will be added separately; their per-sample predictions are already included.

## Repository layout

```
code/
  experiments/            training / inference (GPU)
    check_paired.py           verify that a split is PrimeVul Paired (50/50, adjacent vulnerable/patched pairs)
    run_llm_config.py         CodeLlama-7B / DeepSeek-Coder-6.7B-Instruct: ZS, FS-2, LoRA FT (YES/NO probability scoring)
    run_transformer_seed.py   CodeBERT / GraphCodeBERT / UniXcoder, one seed per run
    aggregate_seeds.py        mean ± std over seeds
  analysis/               CPU only, runs on the files in results/
    build_table3.py           Table 3 (incl. P-C) and the LLM pairwise binomial tests (Section 4.1)
    cwe_deficit.py            Table 5, CWE-617 permutation / max-statistic / CVE-level tests, bootstrap,
                              TensorFlow split, robustness check (Section 4.2.2)
    pair_level.py             pair-level table and test (Section 4.2.4)
    common.py                 shared loaders
  figures/
    make_fig6.py              Figure 6 (lexical similarity vs. detection rate)
    make_fig8.py              Figure 8 (CodeBERT attention, CWE-617)
results/
  test_index.csv            the 868 test samples: idx, target, cwe, cve (no source code)
  rq2_predictions/          original-run predictions of the 17 configurations used in RQ2/RQ3
  llm/                      re-run LLM configurations (predictions, metrics, fine-tuning logs)
  transformer_seeds/        transformer runs, seeds 42 / 1 / 2 (predictions, metrics, summary)
  analysis/                 exp31 (per-CWE similarity vs. difficulty), exp34 (CWE-617 attention tokens)
figures/                  Figures 6 and 8 as used in the paper
data/                     place the PrimeVul Paired CSVs here (not included, see below)
```

## Data

The PrimeVul dataset is **not** redistributed here. Obtain PrimeVul Paired from its authors
(Ding et al., *Vulnerability Detection with Code Language Models: How Far Are We?*, ICSE 2025) and export the
train / validation / test splits as CSV files with the columns `idx, func, target, cwe, cve`, rows ordered so that
each vulnerable function is immediately followed by its patched version. `results/test_index.csv` lists the exact
868 test samples (by PrimeVul `idx`) used in the paper. Check a split with:

```bash
python code/experiments/check_paired.py data/train.csv data/valid.csv data/test.csv
```

## Reproducing the analyses (CPU, no data needed)

```bash
pip install -r requirements.txt
python code/analysis/build_table3.py      # Table 3 + LLM pairwise tests
python code/analysis/cwe_deficit.py       # Table 5 + CWE-617 statistics (~1-2 min)
python code/analysis/pair_level.py        # pair-level table
python code/figures/make_fig6.py results/analysis/exp31_similarity_vs_difficulty.csv figures/exp39_similarity_scatter.png
python code/figures/make_fig8.py results/analysis/exp34_cwe617_attention.csv figures/exp40_attention_heatmap.png
```

Derived tables are written to `results/derived/`. Permutation p-values use a fixed random seed (42).

## Re-running the experiments (GPU)

All runs used a single NVIDIA A100 (40 GB).

```bash
# LLMs (one configuration per call; zs | fs2 | ft)
python code/experiments/run_llm_config.py --config codellama_zs --data_dir data --out_dir results/llm
python code/experiments/run_llm_config.py --config deepseek_ft  --data_dir data --out_dir results/llm \
       --epochs 3 --lr 1e-4 --grad_accum 32 --select auc

# Transformers, three seeds each
for m in codebert graphcodebert unixcoder; do for s in 42 1 2; do
  python code/experiments/run_transformer_seed.py --model $m --seed $s --data_dir data --tag primevul --select auc
done; done
python code/experiments/aggregate_seeds.py seed_results
```

Settings : transformers use AdamW, lr 2e-5, effective batch 32, up to 10 epochs,
10% warmup, early stopping (patience 5), best validation AUC. LLM fine-tuning uses LoRA (r = 16, α = 32,
dropout 0.05, q/v projections), lr 1e-4, up to 3 epochs, best validation AUC. LLM scores are the normalised
probability of YES- vs. NO-variant answer tokens; functions are truncated to 1,500 tokens.

## Notes

- The CWE-level analyses use the original-run predictions in `results/rq2_predictions/`
  (LLM ZS/FS-2 labels there were parsed from free-text answers). `cwe_deficit.py` also repeats the analysis
  with the Table 3 runs as a robustness check.
- The transformer and fine-tuned LLM runs used a reload of the snapshot with 7,578 / 960 train / validation
  samples; ML/DL runs used 7,574 / 956. The test set is identical for all models.

## License

Code is released under the MIT License (see `LICENSE`). Results files are provided for research use.
PrimeVul is subject to its original license.
