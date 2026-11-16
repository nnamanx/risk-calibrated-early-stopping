<div align="center">

# Risk-Calibrated Early Stopping for LLM Training Recipes

**Schedule-aware learning curve prediction with finite-sample winner protection.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX / Flax](https://img.shields.io/badge/JAX-Flax%20%7C%20Optax-008080?logo=google&logoColor=white)](https://github.com/google/flax)
[![Dataset: TinyStories](<https://img.shields.io/badge/Dataset-TinyStories%20(Pinned)-FFD21E?logo=huggingface&logoColor=black>)](https://huggingface.co/datasets/roneneldan/TinyStories)
[![Status: Week 1 Verified](https://img.shields.io/badge/Status-Pipeline%20Verified-2EA44F)](#current-progress)

</div>

---

## Overview

Searching over language-model training recipes (peak learning rate, warmup duration, decay schedule, and weight decay) is expensive, yet standard early-stopping heuristics frequently eliminate **late-blooming winners**—recipes with longer warmups or late-dropping cosine schedules that look suboptimal early on.

This repository implements a reproducible research pipeline to test whether **schedule-aware final-loss prediction** paired with **trajectory- and pool-calibrated conformal lower bounds** can cut recipe-search compute while keeping the probability of eliminating the eventual best recipe below a user-specified risk target ($\alpha = 0.10$).

### Key Research Questions

1. **Schedule-Aware Prediction:** Do known future learning-rate schedule features improve final-loss prediction beyond early loss and slope alone?
2. **Whole-Pool Protection:** Does accounting for repeated decision checkpoints and candidate pool size ($M=7$) prevent false eliminations compared to naive thresholds and existing baselines (Successive Halving, LC-PFN, CBES)?
3. **Calibration Economics:** When does the upfront cost of collecting calibration trajectories pay off across repeated hyperparameter searches?

---

## Experimental Protocol & Architecture

All core experiments isolate training-recipe variables by locking the model architecture, tokenizer, data stream, and training horizon:

| Component              | Locked Specification                                                                                                                                            |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model Architecture** | **5.31M** parameter Llama-like decoder (`5,310,720` exact): 4 layers, width `256`, 8 heads, SwiGLU (`704`), RoPE, RMSNorm, tied embeddings                      |
| **Dataset & Splits**   | `roneneldan/TinyStories` (pinned rev `f54c09fd`), deduplicated into disjoint **Train** (`1.80M` docs), **Monitor** (`7,555` docs), and **Audit** (`7,834` docs) |
| **Tokenizer**          | Custom 8,192-token SentencePiece BPE trained strictly on `10 MB` of training text                                                                               |
| **Training Horizon**   | Fixed **2,048 updates** (`33.55M` target tokens; effective batch `64 × 256`), AdamW (`β₁ = 0.9, β₂ = 0.95`), FP32                                               |
| **Decision Grid**      | Monitor evaluation every `128` steps; irreversible pruning allowed only at updates `{256, 512, 1024, 1536}`                                                     |
| **Search Space**       | Peak LR ~ `LogUniform[1e-4, 2e-3]`, Warmup ∈ `{2%, 8%}`, Decay ∈ `{Cosine 10%, Linear 0}`, WD ∈ `{0, 0.1}`                                                      |

```text
[Pinned TinyStories] ──► [Deduplicated Splits + 8k BPE] ──► [5.31M Llama-JAX Trajectories (2,048 steps)]
                                                                                │
[Offline Replay & Risk Eval] ◄── [Conformal Lower Bound L(i,k)] ◄── [Prefix + Future Schedule Features]
```

## 📂 Repository Structure

```text
thesis-risk-early-stopping/
├── configs/                 # Versioned YAML recipe manifests (pilot, dev, calibration, locked test)
├── manifests/               # Frozen dataset split metadata, SHA-256 hashes, & SentencePiece model
│   └── tinystories/
├── notebooks/               # Thin Google Colab GPU launchers & throughput benchmarks
├── src/
│   ├── training/            # 5.31M Llama-JAX model, data prep, AdamW loop, & atomic checkpointing
│   ├── features/            # Prefix curve & known future schedule feature extraction
│   ├── policies/            # Conformal lower-bound policy & baselines (SH, LC-PFN, CBES)
│   └── replay/              # Cost-ledger simulator (uncapped & 25%–100% fixed-budget replay)
├── tests/                   # Automated leakage, masking, & replay correctness tests
├── results/                 # Archived trajectory tables & publication figures
└── thesis/                  # LaTeX manuscript, literature matrix, & decisions log
```

## Citation & License

- **Dataset License:** TinyStories is used under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
- **Reproducibility Note:** Raw data shards and large model checkpoints are excluded from Git via `.gitignore` and verified via SHA-256 manifests in `manifests/tinystories/dataset_manifest.json`.
