import json
import os
import tempfile
import time
from pathlib import Path
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.training.model import LlamaDecoderLM, ModelConfig, count_parameters


def make_weight_decay_mask(params):
    """Exclude 1D parameters (RMSNorm scales) from weight decay."""
    def _mask(path, leaf):
        key_names = [str(k) for k in path]
        if any("norm" in k.lower() for k in key_names):
            return False
        return True

    return jax.tree_util.tree_map_with_path(_mask, params)


def create_optimizer(peak_lr: float, weight_decay: float, total_steps: int, warmup_steps: int):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.1 * peak_lr,
    )
    return schedule, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
            mask=make_weight_decay_mask,
        ),
    )


def save_checkpoint_atomic(ckpt_path: Path, state_dict: dict) -> int:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = serialization.to_bytes(state_dict)
    with tempfile.NamedTemporaryFile(dir=ckpt_path.parent, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, ckpt_path)
    return ckpt_path.stat().st_size


def load_checkpoint(ckpt_path: Path, template_state_dict: dict) -> dict:
    with open(ckpt_path, "rb") as f:
        raw_bytes = f.read()
    return serialization.from_bytes(template_state_dict, raw_bytes)


def main():
    cfg = ModelConfig()
    model = LlamaDecoderLM(cfg)

    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((2, cfg.context_length), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    n_params = count_parameters(params)
    print(f"Initialized LlamaDecoderLM | Exact parameter count: {n_params:,}")
    assert n_params == 5_310_720, f"Expected 5,310,720 params, got {n_params:,}"

    # Load a tiny slice of the frozen TinyStories train shard (or fallback synthetic tokens)
    train_shard = Path("data/processed/tinystories/train_tokens.npy")
    if train_shard.exists():
        tokens = np.load(train_shard)[: 64 * 257]
    else:
        tokens = np.random.default_rng(42).integers(
            0, cfg.vocab_size, size=64 * 257, dtype=np.uint16)

    windows = tokens.reshape(-1, cfg.context_length + 1).astype(np.int32)
    inputs = jnp.array(windows[:16, :-1])   # (16, 256)
    targets = jnp.array(windows[:16, 1:])   # (16, 256) shifted by 1 token

    total_steps = 2048
    warmup_steps = int(np.ceil(0.02 * total_steps))  # 41 updates
    schedule, tx = create_optimizer(
        peak_lr=6e-4, weight_decay=0.1, total_steps=total_steps, warmup_steps=warmup_steps
    )
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            logits = model.apply(p, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, y).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grad_norm

    # Run 60 debug updates
    losses = []
    t0 = time.perf_counter()
    for step in range(1, 61):
        params, opt_state, loss, gnorm = train_step(
            params, opt_state, inputs, targets)
        loss_val = float(loss)
        losses.append(loss_val)
        if step in {1, 10, 20, 40, 60}:
            lr_now = float(schedule(step - 1))
            print(
                f"Step {step:02d}/60 | lr={lr_now:.6f} | loss={loss_val:.4f} | grad_norm={float(gnorm):.3f}")

    elapsed = time.perf_counter() - t0
    print(
        f"Completed 60 debug updates in {elapsed:.2f}s | Initial loss: {losses[0]:.4f} -> Final loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Debug check failed: loss did not decrease!"

    # Test atomic checkpoint save & reload equivalence
    ckpt_dir = Path("checkpoints/debug_check")
    ckpt_file = ckpt_dir / "step_0060.msgpack"
    state_payload = {"step": 60, "params": params, "opt_state": opt_state}
    ckpt_bytes = save_checkpoint_atomic(ckpt_file, state_payload)
    print(f"Saved atomic checkpoint ({ckpt_bytes:,} bytes) to {ckpt_file}")

    restored = load_checkpoint(ckpt_file, state_payload)
    assert restored["step"] == 60

    # Compare step 61 from uninterrupted vs restored state
    _, _, loss_uninterrupted, _ = train_step(
        params, opt_state, inputs, targets)
    _, _, loss_restored, _ = train_step(
        restored["params"], restored["opt_state"], inputs, targets)
    diff = abs(float(loss_uninterrupted) - float(loss_restored))
    print(f"Step 61 loss diff after reload: {diff:.2e}")
    assert diff < 1e-6, f"Checkpoint reload mismatch: {diff}"
    print("ALL DEBUG CHECKS PASSED.")


if __name__ == "__main__":
    main()
