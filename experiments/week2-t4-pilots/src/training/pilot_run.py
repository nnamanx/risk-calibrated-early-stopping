import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.training.debug_run import (
    load_checkpoint,
    make_weight_decay_mask,
    save_checkpoint_atomic,
)
from src.training.model import LlamaDecoderLM, ModelConfig, count_parameters


TOTAL_STEPS = 2048
CONTEXT_LENGTH = 256
MICROBATCH_SIZE = 16
MICROBATCHES_PER_UPDATE = 4
BATCH_SIZE = MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE
TOKENS_PER_UPDATE = BATCH_SIZE * CONTEXT_LENGTH
MONITOR_EVERY = 128
CHECKPOINT_STEPS = (256, 512, 1024, 1536, TOTAL_STEPS)

DATA_DIR = Path("data/processed/tinystories")
MANIFEST_PATH = Path("manifests/tinystories/dataset_manifest.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data():
    manifest = json.loads(MANIFEST_PATH.read_text())
    streams = manifest["token_streams"]
    tokenizer = manifest["tokenizer"]

    expected_hashes = {
        DATA_DIR / "train_tokens.npy": streams["train_shard_sha256"],
        DATA_DIR / "monitor_windows.npy": streams["monitor_sha256"],
        DATA_DIR / "audit_windows.npy": streams["audit_sha256"],
        Path(tokenizer["model_path"]): tokenizer["model_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"File differs from the Week 1 manifest: {path}")

    train = np.load(DATA_DIR / "train_tokens.npy", mmap_mode="r")
    monitor = np.load(DATA_DIR / "monitor_windows.npy")
    audit = np.load(DATA_DIR / "audit_windows.npy", mmap_mode="r")

    if train.ndim != 1 or train.size < TOTAL_STEPS * TOKENS_PER_UPDATE + 1:
        raise ValueError("Training stream is too short for the full pilot.")
    if monitor.shape != (1024, 257) or audit.shape != (4096, 257):
        raise ValueError("Unexpected monitor or audit window shape.")

    identity = {
        "dataset_revision": manifest["dataset_revision"],
        "dataset_hash": sha256_file(MANIFEST_PATH),
        "train_tokens_hash": streams["train_shard_sha256"],
        "tokenizer_hash": tokenizer["model_sha256"],
        "monitor_windows": len(monitor),
        "audit_windows": len(audit),
    }
    return train, monitor, identity


def build_optimizer(recipe):
    peak = recipe["peak_lr"]
    warmup_steps = math.ceil(recipe["warmup"] * TOTAL_STEPS)
    decay_steps = TOTAL_STEPS - warmup_steps
    warmup = optax.linear_schedule(0.0, peak, warmup_steps)

    if recipe["schedule"] == "cosine":
        decay = optax.cosine_decay_schedule(peak, decay_steps, alpha=0.1)
    elif recipe["schedule"] == "linear":
        decay = optax.linear_schedule(peak, 0.0, decay_steps)
    else:
        raise ValueError(f"Unknown schedule: {recipe['schedule']}")

    schedule = optax.join_schedules([warmup, decay], [warmup_steps])
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=recipe["weight_decay"],
            mask=make_weight_decay_mask,
        ),
    )
    return schedule, optimizer


def run_pilot(recipe, output_dir, stop_at):
    session_start = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Use a JAX GPU runtime for the full pilot.")

    # 1. Load the same frozen data for every recipe.
    train, monitor, identity = load_data()
    monitor = jnp.asarray(monitor, dtype=jnp.int32)

    # 2. Build the model and the optimizer for this recipe.
    model = LlamaDecoderLM(ModelConfig())
    key = jax.random.PRNGKey(recipe["seed"])
    dummy_input = jnp.zeros((1, CONTEXT_LENGTH), dtype=jnp.int32)
    params = model.init(key, dummy_input)
    parameter_count = count_parameters(params)
    if parameter_count != 5_310_720:
        raise ValueError(f"Unexpected parameter count: {parameter_count}")

    schedule, optimizer = build_optimizer(recipe)
    opt_state = optimizer.init(params)

    def batch_loss(weights, windows):
        inputs = windows[:, :-1]
        targets = windows[:, 1:]
        logits = model.apply(weights, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()

    loss_and_grad = jax.value_and_grad(batch_loss)
    evaluate_batch = jax.jit(batch_loss)

    # 3. Average four microbatch gradients before changing the weights.
    @jax.jit
    def train_step(weights, state, microbatches):
        def accumulate(index, carry):
            loss_sum, gradient_sum = carry
            loss, gradients = loss_and_grad(weights, microbatches[index])
            gradient_sum = jax.tree_util.tree_map(
                jnp.add, gradient_sum, gradients
            )
            return loss_sum + loss, gradient_sum

        zero_gradients = jax.tree_util.tree_map(jnp.zeros_like, weights)
        loss_sum, gradient_sum = jax.lax.fori_loop(
            0, MICROBATCHES_PER_UPDATE, accumulate,
            (jnp.float32(0), zero_gradients),
        )
        mean_gradients = jax.tree_util.tree_map(
            lambda value: value / MICROBATCHES_PER_UPDATE, gradient_sum
        )
        updates, next_state = optimizer.update(mean_gradients, state, weights)
        next_weights = optax.apply_updates(weights, updates)
        return (
            next_weights,
            next_state,
            loss_sum / MICROBATCHES_PER_UPDATE,
            optax.global_norm(mean_gradients),
        )

    def monitor_loss(weights):
        losses = []
        for start in range(0, len(monitor), MICROBATCH_SIZE):
            batch = monitor[start:start + MICROBATCH_SIZE]
            losses.append(float(evaluate_batch(weights, batch)))
        return float(np.mean(losses))

    # 4. Find the saved state, if this is a resumed run.
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metrics_path = output_dir / "metrics.jsonl"
    checkpoints = sorted(output_dir.glob("step_*.msgpack"))

    metadata = {
        **identity,
        "recipe": recipe,
        "total_steps": TOTAL_STEPS,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "parameter_count": parameter_count,
        "precision": "FP32",
        "gpu": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "script_hash": sha256_file(Path(__file__)),
        "model_code_hash": sha256_file(Path("src/training/model.py")),
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Run metadata changed. Use a new output folder.")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    restore_seconds = 0.0
    state = {
        "step": 0,
        "params": params,
        "opt_state": opt_state,
        "train_seconds": 0.0,
        "monitor_seconds": 0.0,
    }
    if checkpoints:
        restore_start = time.perf_counter()
        state = load_checkpoint(checkpoints[-1], state)
        restore_seconds = time.perf_counter() - restore_start

    start_step = state["step"]
    if start_step >= stop_at:
        print(f"{recipe['run_id']} already reached step {start_step}.")
        return

    # Keep only rows that belong to the saved state when restarting.
    retained = []
    if start_step:
        for line in metrics_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["step"] <= start_step:
                retained.append(row)
        if [row["step"] for row in retained] != list(range(start_step + 1)):
            raise ValueError("Saved checkpoint and metrics history disagree.")
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in retained))

    params = state["params"]
    opt_state = state["opt_state"]
    train_seconds = state["train_seconds"]
    monitor_seconds = state["monitor_seconds"]
    checkpoint_seconds = 0.0
    checkpoint_bytes = 0
    row_starts = np.arange(BATCH_SIZE)[:, None] * CONTEXT_LENGTH
    offsets = row_starts + np.arange(CONTEXT_LENGTH + 1)

    print(f"Run: {recipe['run_id']} | GPU: {metadata['gpu']}", flush=True)
    print(
        f"Starting at step {start_step}, running through {stop_at}", flush=True)

    # 5. Train, check monitor loss, and save progress.
    first_step = start_step + 1 if checkpoints else 0
    with metrics_path.open("a") as log:
        for step in range(first_step, stop_at + 1):
            loss = None
            gradient_norm = None
            learning_rate = None

            if step:
                train_start = time.perf_counter()
                batch_start = (step - 1) * TOKENS_PER_UPDATE
                windows = train[batch_start + offsets]
                microbatches = jnp.asarray(windows, dtype=jnp.int32).reshape(
                    MICROBATCHES_PER_UPDATE, MICROBATCH_SIZE, CONTEXT_LENGTH + 1
                )
                params, opt_state, loss, gradient_norm = jax.block_until_ready(
                    train_step(params, opt_state, microbatches)
                )
                train_seconds += time.perf_counter() - train_start
                loss, gradient_norm = float(loss), float(gradient_norm)
                learning_rate = float(schedule(step - 1))
                if not math.isfinite(loss) or not math.isfinite(gradient_norm):
                    raise RuntimeError(
                        f"Non-finite training value at step {step}")

            validation_loss = None
            if step % MONITOR_EVERY == 0:
                monitor_start = time.perf_counter()
                validation_loss = monitor_loss(params)
                monitor_seconds += time.perf_counter() - monitor_start
                if not math.isfinite(validation_loss):
                    raise RuntimeError(
                        f"Non-finite monitor loss at step {step}")
                print(
                    f"Step {step:4d}/{TOTAL_STEPS} | monitor loss {validation_loss:.4f}",
                    flush=True,
                )

            row = {
                "step": step,
                "tokens_seen": step * TOKENS_PER_UPDATE,
                "learning_rate": learning_rate,
                "train_loss": loss,
                "gradient_norm": gradient_norm,
                "monitor_loss": validation_loss,
                "train_seconds": train_seconds,
                "monitor_seconds": monitor_seconds,
                "dataset_hash": identity["dataset_hash"],
                "tokenizer_hash": identity["tokenizer_hash"],
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()

            if step in CHECKPOINT_STEPS:
                checkpoint_start = time.perf_counter()
                checkpoint_bytes = save_checkpoint_atomic(
                    output_dir / f"step_{step:04d}.msgpack",
                    {
                        "step": step,
                        "params": params,
                        "opt_state": opt_state,
                        "train_seconds": train_seconds,
                        "monitor_seconds": monitor_seconds,
                    },
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_start

    # 6. Save one summary for this session, including pause or restart costs.
    summary = {
        **metadata,
        "status": "completed" if stop_at == TOTAL_STEPS else "paused",
        "termination_reason": "full_horizon" if stop_at == TOTAL_STEPS
        else "planned_pause",
        "restored_from_step": start_step,
        "last_step": stop_at,
        "tokens_seen": stop_at * TOKENS_PER_UPDATE,
        "final_train_loss": loss,
        "final_monitor_loss": validation_loss,
        "train_seconds": train_seconds,
        "monitor_seconds": monitor_seconds,
        "restore_seconds": restore_seconds,
        "checkpoint_seconds_this_session": checkpoint_seconds,
        "last_checkpoint_bytes": checkpoint_bytes,
        "session_wall_seconds": time.perf_counter() - session_start,
    }
    summary_path = output_dir / f"summary_{stop_at:04d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/pilot_01.json"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/pilots"))
    parser.add_argument("--pause-after", type=int,
                        choices=CHECKPOINT_STEPS[:-1])
    args = parser.parse_args()

    recipe = json.loads(args.config.read_text())
    output_dir = args.output_root / recipe["run_id"]
    run_pilot(recipe, output_dir, args.pause_after or TOTAL_STEPS)


if __name__ == "__main__":
    main()
