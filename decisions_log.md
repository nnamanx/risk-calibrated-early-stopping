# Research & Engineering Decisions Log

## Format
- **Date:** YYYY-MM-DD
- **Decision ID:** DEC-XXX
- **Category:** Scope | Data | Model | Method | Compute | Evaluation
- **Decision:** Clear statement of what was decided.
- **Rationale & Evidence:** Why this choice was made and what data/literature supports it.
- **Impact / Frozen Artifacts:** What files, manifests, or protocols are locked by this decision.

---

## Initial Baseline Entries

### DEC-001: Core Study Scope & Target Risk Definition
- **Date:** 2026-11-16
- **Category:** Scope
- **Decision:** Focus the core study on training-recipe early stopping for a ~5.31M parameter Llama-like decoder trained from scratch over a fixed 2,048-update (33,554,432-token) horizon, evaluated in 8-candidate pools (1 anchor + 7 uncertain candidates) with primary voluntary-elimination risk target $\alpha = 0.10$ and near-best margin $1.01 \times y_{\text{best}}$.
- **Rationale & Evidence:** Isolates learning-rate, warmup, decay schedule, and weight-decay dynamics under a strictly controlled compute budget before scaling to 17.31M parameters or domain transfer.
- **Impact / Frozen Artifacts:** `protocol.md`, `configs/`

### DEC-002: Dataset Priority & Fallback Rule
- **Date:** 2026-11-16
- **Category:** Data
- **Decision:** Evaluate FineWeb-Edu subset vs. TinyStories cost during the 12-run pilot. Use `roneneldan/TinyStories` (8,192 BPE SentencePiece tokenizer fit on 10 MB of deduplicated train text; 1,024 monitor windows; 4,096 audit windows) as the main dataset if FineWeb-Edu exceeds available Colab compute, reserving FineWeb-Edu for a 24-run domain-transfer stress test.
- **Rationale & Evidence:** Ensures completion of the full 198-run core matrix (or 106-run fallback) within the pre-Dec 27 compute schedule.
- **Impact / Frozen Artifacts:** `manifests/`

### DEC-003: Run & Checkpoint Logging Identifiers
- **Date:** 2026-11-16
- **Category:** Compute
- **Decision:** Enforce logging of all 13 run-level identifiers (`run_id`, `recipe_id`, `split`, `configuration_hash`, `code_revision`, `package_versions`, `dataset_hash`, `tokenizer_hash`, `seed`, `gpu_type`, `precision`, `parameter_count`, `fixed_training_horizon`), 9 checkpoint-level metrics (`optimizer_step`, `tokens_seen`, `current_learning_rate`, `training_loss`, `monitor_loss`, `gradient_norm`, `elapsed_training_seconds`, `evaluation_seconds`, `checkpoint_bytes`), and separate `termination_reason` vs. `completion_status` fields.
- **Rationale & Evidence:** Required for exact cost accounting, future-leakage prevention during replay, and distinguishing voluntary policy pruning from budget censoring or infrastructure faults.
- **Impact / Frozen Artifacts:** `run_register.csv`, `src/training/`
