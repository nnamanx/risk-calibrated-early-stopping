import time
import subprocess
import json
import os
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from src.training.model import LlamaDecoderLM, ModelConfig


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        return out
    except Exception:
        return str(jax.devices())


def get_peak_gpu_memory_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip()
        return max(int(x.strip()) for x in out.splitlines())
    except Exception:
        return -1


def main():
    print("=== JAX & Hardware Environment ===")
    print(f"JAX version: {jax.__version__} | Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    gpu_info = get_gpu_info()
    print(f"GPU Info: {gpu_info}")

    # Locked protocol settings: effective batch = 64 x 256 = 16,384 tokens
    # Microbatch = 16, grad accumulation = 4
    micro_batch = 16
    accum_steps = 4
    seq_len = 256
    tokens_per_update = micro_batch * accum_steps * seq_len  # 16,384
    horizon_updates = 2048
    tokens_per_run = horizon_updates * tokens_per_update     # 33,554,432

    cfg = ModelConfig()
    model = LlamaDecoderLM(cfg)
    rng = jax.random.PRNGKey(42)
    dummy_x = jnp.ones((micro_batch, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_x)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Exact Parameter Count: {param_count:,} | Precision: FP32")

    # Optimizer with gradient accumulation (4 micro-steps per update)
    base_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=6e-4, b1=0.9,
                    b2=0.95, eps=1e-8, weight_decay=0.1)
    )
    optimizer = optax.MultiSteps(base_opt, every_k_schedule=accum_steps)
    opt_state = optimizer.init(params)

    def loss_fn(p, x, y):
        logits = model.apply(p, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, y).mean()
        return loss

    @jax.jit
    def micro_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def eval_step(params, x, y):
        return loss_fn(params, x, y)

    # Synthetic microbatches (micro_batch, seq_len)
    x_data = jax.random.randint(
        rng, (accum_steps, micro_batch, seq_len), 0, cfg.vocab_size)
    y_data = jax.random.randint(
        rng, (accum_steps, micro_batch, seq_len), 0, cfg.vocab_size)

    # 1. Warmup compilation (excluded from timing)
    print("\nCompiling XLA kernels (warmup)...")
    t_compile_0 = time.perf_counter()
    for _ in range(2):
        for m in range(accum_steps):
            params, opt_state, loss = micro_step(
                params, opt_state, x_data[m], y_data[m])
    jax.block_until_ready(loss)
    compile_sec = time.perf_counter() - t_compile_0
    print(f"Compilation + 2 warmup updates finished in {compile_sec:.2f}s")

    # 2. Synchronized 200 optimizer updates (800 micro-steps)
    num_updates = 200
    print(
        f"\nTiming {num_updates} synchronized optimizer updates ({num_updates * tokens_per_update:,} tokens)...")
    jax.block_until_ready(loss)
    t0 = time.perf_counter()
    for _ in range(num_updates):
        for m in range(accum_steps):
            params, opt_state, loss = micro_step(
                params, opt_state, x_data[m], y_data[m])
    jax.block_until_ready(loss)
    train_sec = time.perf_counter() - t0

    tokens_benchmarked = num_updates * tokens_per_update
    tokens_per_sec = tokens_benchmarked / train_sec
    sec_per_update = train_sec / num_updates
    est_train_sec_per_run = sec_per_update * horizon_updates

    # 3. Measure monitoring evaluation (1,024 windows = 64 batches of 16)
    eval_batches = 1024 // micro_batch
    _ = eval_step(params, x_data[0], y_data[0]).block_until_ready()
    t_eval_0 = time.perf_counter()
    for _ in range(eval_batches):
        eval_loss = eval_step(params, x_data[0], y_data[0])
    jax.block_until_ready(eval_loss)
    eval_sec_once = time.perf_counter() - t_eval_0
    # 17 monitor evals per run (steps 0, 128, 256, ..., 2048)
    est_eval_sec_per_run = eval_sec_once * 17

    # 4. Measure checkpoint serialization & disk write
    os.makedirs("checkpoints/bench_tmp", exist_ok=True)
    ckpt_path = "checkpoints/bench_tmp/test_ckpt.msgpack"
    t_ckpt_0 = time.perf_counter()
    payload = serialization.to_bytes(
        {"params": params, "opt_state": opt_state})
    with open(ckpt_path, "wb") as f:
        f.write(payload)
    ckpt_sec_once = time.perf_counter() - t_ckpt_0
    ckpt_mb = len(payload) / (1024 * 1024)
    # 4 decision checkpoints + final checkpoint = 5 saves per run
    est_ckpt_sec_per_run = ckpt_sec_once * 5
    os.remove(ckpt_path)

    peak_mem_mb = get_peak_gpu_memory_mb()
    total_sec_per_run = compile_sec + est_train_sec_per_run + \
        est_eval_sec_per_run + est_ckpt_sec_per_run
    total_hrs_per_run = total_sec_per_run / 3600.0

    # Budget projections with 30% operational reserve
    hrs_198_core = total_hrs_per_run * 198 * 1.30
    hrs_106_small = total_hrs_per_run * 106 * 1.30

    print("\n================ BENCHMARK RESULTS ================")
    print(f"GPU Hardware          : {gpu_info}")
    print(f"Peak GPU Memory       : {peak_mem_mb} MB")
    print(
        f"Throughput            : {tokens_per_sec:,.0f} tokens/sec ({sec_per_update*1000:.2f} ms/update)")
    print(
        f"1x Monitor Eval (1024): {eval_sec_once:.3f}s (17x/run = {est_eval_sec_per_run:.1f}s)")
    print(
        f"1x Checkpoint ({ckpt_mb:.1f} MB): {ckpt_sec_once:.3f}s (5x/run = {est_ckpt_sec_per_run:.1f}s)")
    print(
        f"Est. Full Run (2,048) : {total_sec_per_run/60:.2f} mins ({total_hrs_per_run:.3f} hrs)")
    print("---------------------------------------------------")
    print(f"198-Run Core (+30% reserve)   : {hrs_198_core:.1f} GPU hours")
    print(f"106-Run Smaller (+30% reserve): {hrs_106_small:.1f} GPU hours")
    print("===================================================")

    os.makedirs("results", exist_ok=True)
    summary = {
        "gpu_info": gpu_info,
        "peak_gpu_memory_mb": peak_mem_mb,
        "precision": "FP32",
        "effective_batch_tokens": tokens_per_update,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "ms_per_update": round(sec_per_update * 1000, 2),
        "compile_sec": round(compile_sec, 2),
        "eval_sec_once": round(eval_sec_once, 3),
        "ckpt_sec_once": round(ckpt_sec_once, 3),
        "est_minutes_per_run": round(total_sec_per_run / 60, 2),
        "est_hours_198_core_with_reserve": round(hrs_198_core, 1),
        "est_hours_106_small_with_reserve": round(hrs_106_small, 1),
    }
    with open("results/week1_gpu_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results to results/week1_gpu_benchmark.json")


if __name__ == "__main__":
    main()
