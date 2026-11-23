# Week 2 TinyStories T4 pilots

This experiment ran 12 training recipes for 2,048 updates each on a Tesla T4.

See [pilot results](RESULTS.md) for the findings and their implications for early stopping.

- `configs/` contains the 12 recipes.
- `results/` contains each run's metadata, step-by-step metrics, and final summary.
- The dataset manifest records the inputs used for the experiment.

## Reproduce a run

Use Git revision `b8f4b5e52aac85cbb8c13685e2b9ace54433b31b` for the
original source. The pilot runner is `src/training/pilot_run.py` at the
repository root.

The pilots used Google Colab with a Tesla T4, Python 3.13, JAX 0.11.1,
Flax 0.12.12, and Optax 0.2.8. Check that `jax.devices()` shows a CUDA device.

Restore the original TinyStories arrays from `thesis-pilot-01.zip` in
Google Drive. Place these three files under `data/processed/tinystories/`
relative to the repository root:

- `train_tokens.npy`
- `monitor_windows.npy`
- `audit_windows.npy`

The runner checks the input files against the hashes in the top-level
dataset manifest. From the repository root, run:

```bash
python -u -m src.training.pilot_run \
  --config experiments/week2-t4-pilots/configs/pilot_01.json \
  --output-root results/pilots-reproduction
```

To reproduce the other recipes, change `pilot_01.json` to `pilot_02.json`
through `pilot_12.json`. The reproduction command writes to a separate
output directory so the archived results can be used for comparison.
