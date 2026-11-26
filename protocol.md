# TinyStories early-stopping protocol

**Study:** 198-run design. **Recorded:** 2026-11-26, after development began. The design below records the launch setup and prior decisions. The exact saved Colab manifest and trainer revision still need verification before signing off on the as-run protocol.

## Training configuration

| Setting | Value |
| --- | --- |
| Model and data | Randomly initialized 5.31M-parameter decoder-only Transformer trained on TinyStories with causal next-token loss and context length 256. |
| Optimizer | AdamW with β₁ = 0.9, β₂ = 0.95, ε = 1e-8, and global gradient norm clipping at 1.0. Weight decay is recipe-specific and uses the trainer's parameter mask. |
| Precision | FP32 in the documented T4 setup. Confirm against the core run metadata. |
| Token horizon | 2,048 optimizer updates × 64 sequences × 256 target tokens = 33,554,432 target tokens per full run. Each update accumulates four microbatches of 16 sequences. |
| Monitor evaluation | Every 128 updates on the held-out TinyStories monitor windows. |
| Saved checkpoints | Updates 256, 512, 1,024, 1,536, and 2,048. |
| Stopping decisions | Updates 256, 512, 1,024, and 1,536. Update 2,048 supplies the full-run comparison loss. Use the schedule defined for the full horizon for every candidate. |

## Candidate distribution

The Colab manifest-generation cell uses sampling seed `20261124`. The 184 candidate recipes in the development, calibration, and locked-test splits use the following distribution.

| Recipe field | Sampling rule |
| --- | --- |
| Peak learning rate | Log-uniform from 1e-4 to 2e-3. |
| Warmup | Uniform choice of 0.02 or 0.08 of the full horizon, rounded up to 41 or 164 updates. |
| Decay schedule | Uniform choice of cosine decay to 10% of peak learning rate or linear decay to zero. |
| Weight decay | Uniform choice of 0 or 0.1. |
| Training seed | Distinct seeds recorded in the manifest. Anchor runs use fresh seeds. |

## Run inventory and evaluation pools

| Run category | Count |
| --- | ---: |
| Pilots | 12 |
| Development candidates | 48 |
| Calibration candidates | 80 |
| Locked-test candidates | 56 |
| Anchors | 2 |
| Total | 198 |

The saved launch manifest is `MyDrive/thesis-core-198-t4/manifest.json`. The launcher writes per-run configurations under `configs/` and run outputs under `results/` in the same study folder. The 12 pilots are tracked separately and excluded from development, calibration, and locked-test pools.

Use development outcomes to select the anchor recipe. Train a development anchor and a held-out evaluation anchor using fresh, distinct seeds. The primary analysis uses eight candidates per pool, one fully trained evaluation anchor and seven locked-test candidates. The 56 locked-test candidates form eight disjoint groups of seven, with the evaluation anchor shared across groups. Charge the anchor's full training cost to each simulated search. The descriptive four-candidate analysis uses one anchor and three candidates per pool.

## Near-best outcome

For each evaluation pool, let `L_best` be the lowest final monitor loss at update 2,048 among all fully trained pool members, including the anchor. A recipe is near-best if its final monitor loss is at most `1.01 × L_best`. The margin is relative to the pool's best loss. The decisions log sets the primary voluntary-elimination risk target to 0.10.

## Source record and freeze status

- The launch settings above use the recorded Colab Cell 5 and Cell 6 setup. Trainer constants and optimizer settings were cross-checked against `src/training/pilot_run.py` at repository revision `24b91d6aa987fc5bb784bb495ccc3f3ba7128963`.
- The 1% near-best margin, 0.10 risk target, and eight-candidate pool were recorded in `decisions_log.md` (DEC-001, 2026-11-16). The run counts and descriptive four-candidate analysis come from the Week 2 compute decision and thesis roadmap.
- Before marking this protocol frozen, record the saved core manifest's hash and creation time, verify split counts and training seeds, and match the launched trainer revision and precision to the core run metadata. Record when calibration and locked-test outcomes were first inspected. This document was written after development started.
