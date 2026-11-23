from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 256
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 8
    head_dim: int = 32
    d_ff: int = 704
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-
                       1, keepdims=True) + self.eps)
        return (x / rms) * scale


def apply_rope(x: jnp.ndarray, theta: float = 10000.0) -> jnp.ndarray:
    """Apply Rotary Position Embeddings to (batch, seq_len, n_heads, head_dim)."""
    _, seq_len, _, head_dim = x.shape
    half_dim = head_dim // 2
    freqs = 1.0 / \
        (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.einsum("t,d->td", positions, freqs)  # (seq_len, half_dim)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    rotated = jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.astype(x.dtype)


class CausalSelfAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        bsz, seq_len, d_model = x.shape
        n_heads = self.cfg.n_heads
        head_dim = self.cfg.head_dim

        q = nn.Dense(n_heads * head_dim, use_bias=False, name="wq")(x)
        k = nn.Dense(n_heads * head_dim, use_bias=False, name="wk")(x)
        v = nn.Dense(n_heads * head_dim, use_bias=False, name="wv")(x)

        q = q.reshape(bsz, seq_len, n_heads, head_dim)
        k = k.reshape(bsz, seq_len, n_heads, head_dim)
        v = v.reshape(bsz, seq_len, n_heads, head_dim)

        q = apply_rope(q, self.cfg.rope_theta)
        k = apply_rope(k, self.cfg.rope_theta)

        # (bsz, n_heads, seq_len, seq_len)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / \
            jnp.sqrt(float(head_dim))
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[
            None, None, :, :]
        scores = jnp.where(causal_mask, scores, jnp.finfo(scores.dtype).min)

        probs = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bshd->bthd", probs,
                         v).reshape(bsz, seq_len, d_model)
        return nn.Dense(d_model, use_bias=False, name="wo")(out)


class SwiGLUMLP(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.cfg.d_ff, use_bias=False, name="w_gate")(x)
        up = nn.Dense(self.cfg.d_ff, use_bias=False, name="w_up")(x)
        hidden = jax.nn.silu(gate) * up
        return nn.Dense(self.cfg.d_model, use_bias=False, name="w_down")(hidden)


class TransformerBlock(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x + CausalSelfAttention(self.cfg, name="attn")(
            RMSNorm(self.cfg.d_model, self.cfg.norm_eps, name="attn_norm")(x)
        )
        out = h + SwiGLUMLP(self.cfg, name="mlp")(
            RMSNorm(self.cfg.d_model, self.cfg.norm_eps, name="mlp_norm")(h)
        )
        return out


class LlamaDecoderLM(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        embed = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.d_model,
            name="tok_embeddings",
        )
        x = embed(input_ids)
        for i in range(self.cfg.n_layers):
            x = TransformerBlock(self.cfg, name=f"block_{i}")(x)
        x = RMSNorm(self.cfg.d_model, self.cfg.norm_eps, name="final_norm")(x)
        # Tied input/output embeddings
        logits = embed.attend(x)
        return logits


def count_parameters(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
