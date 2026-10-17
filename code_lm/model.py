from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange


LOOP_DEFAULTS = dict(architecture="standard", prelude_layers=0, recurrent_layers=0,
                     coda_layers=0, loop_count=1)


def model_args_from_config(source: dict) -> dict:
    """Read complete architecture metadata; old checkpoints are standard models."""
    names = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    if any(name not in source for name in names):
        raise ValueError("Checkpoint/config is missing model architecture settings")
    if source.get("architecture", "standard") == "loop" and any(k not in source for k in LOOP_DEFAULTS):
        raise ValueError("Loop checkpoints require explicit partition sizes and loop_count")
    return {**{name: source[name] for name in names},
            **{key: source.get(key, default) for key, default in LOOP_DEFAULTS.items()},
            "loop_max_count": source.get("loop_max_count", 0),
            "loop_lora_rank": source.get("loop_lora_rank", 0),
            "loop_lora_alpha": source.get("loop_lora_alpha", 8.0)}


def inference_weights(state: dict, model_args: dict) -> tuple[dict, dict]:
    """Return the next-token model weights and architecture metadata."""
    return state, model_args


class RMSNorm(nn.Module):
    """Root mean square normalization over the final tensor dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.weight
        return output.to(in_dtype)


class SwiGLU(nn.Module):
    """A position-wise gated feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """Apply position-dependent 2D rotations to pairs of feature coordinates."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires an even d_k, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        cosine_cache, sine_cache = self._build_cache(device)
        self.register_buffer("cosine_cache", cosine_cache, persistent=False)
        self.register_buffer("sine_cache", sine_cache, persistent=False)

    def _build_cache(self, device: torch.device | None) -> tuple[Tensor, Tensor]:
        positions = torch.arange(
            self.max_seq_len,
            device=device,
            dtype=torch.float32,
        )
        dimension_indices = torch.arange(
            0,
            self.d_k,
            2,
            device=device,
            dtype=torch.float32,
        )
        frequencies = self.theta ** (-dimension_indices / self.d_k)
        angles = torch.outer(positions, frequencies)

        return angles.cos(), angles.sin()

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        cos = self.cosine_cache[token_positions]
        sin = self.sine_cache[token_positions]
        even, odd = x[..., 0::2], x[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        output = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(start_dim=-2)

        return output.to(in_dtype)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally using RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if (rope_theta is None) != (max_seq_len is None):
            raise ValueError("rope_theta and max_seq_len must be provided together")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        self.rope = (
            RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )
            if rope_theta is not None and max_seq_len is not None
            else None
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None, adapters=None) -> Tensor:
        """Transform ``(..., sequence_length, d_model)`` inputs."""
        seq_len = x.shape[-2]

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        if adapters is not None:
            query = query + adapters['q'](x)
            value = value + adapters['v'](x)

        query = rearrange(query, "... l (h d) -> ... h l d", h=self.num_heads)
        key = rearrange(key, "... l (h d) -> ... h l d", h=self.num_heads)
        value = rearrange(value, "... l (h d) -> ... h l d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                rope_positions = torch.arange(seq_len, device=x.device)
            else:
                rope_positions = token_positions.unsqueeze(-2)

            query = self.rope(query, rope_positions)
            key = self.rope(key, rope_positions)

        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        output = rearrange(output, "... h l d -> ... l (h d)")
        return self.output_proj(output)


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with attention and SwiGLU sublayers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, adapters=None) -> Tensor:
        """Transform ``(..., sequence_length, d_model)`` inputs."""
        y = x + self.attn(self.ln1(x), adapters=adapters)
        return y + self.ffn(self.ln2(y))


class LoopLoRA(nn.Module):
    """One depth-specific Q/V residual; zero output at initialization."""
    def __init__(self, width, rank, alpha, device=None, dtype=None):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, width, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(width, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return F.linear(F.linear(x, self.a), self.b) * self.scale


class TransformerLM(nn.Module):
    """Decoder-only LM with optional fixed-depth weight sharing.

    num_layers counts unique parameterized blocks. In loop mode the middle
    partition is applied loop_count times, with full backpropagation through
    every application. RoPE still encodes token positions, not loop indices.
    Optional MTP adds one independent training head for the second future token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        architecture: str = "standard",
        prelude_layers: int = 0,
        recurrent_layers: int = 0,
        coda_layers: int = 0,
        loop_count: int = 1,
        loop_max_count: int = 0,
        loop_lora_rank: int = 0,
        loop_lora_alpha: float = 8.0,
    ) -> None:
        super().__init__()
        if architecture not in ("standard", "loop"):
            raise ValueError("architecture must be standard or loop")
        sizes = (num_layers, prelude_layers, recurrent_layers, coda_layers, loop_count)
        if any(type(n) is not int for n in sizes) or num_layers < 1 or loop_count < 1:
            raise ValueError("Layer counts must be integers; num_layers and loop_count must be positive")
        if architecture == "standard":
            if (prelude_layers, recurrent_layers, coda_layers, loop_count) != (0, 0, 0, 1):
                raise ValueError("Loop settings require architecture='loop'")
            self.layer_order = tuple(range(num_layers))
        else:
            if min(prelude_layers, coda_layers) < 0 or recurrent_layers < 1:
                raise ValueError("Loop core needs at least one layer; prelude/coda cannot be negative")
            if prelude_layers + recurrent_layers + coda_layers != num_layers:
                raise ValueError("num_layers must equal prelude_layers + recurrent_layers + coda_layers")
            end = prelude_layers + recurrent_layers
            self.layer_order = (tuple(range(prelude_layers))
                                + tuple(range(prelude_layers, end)) * loop_count
                                + tuple(range(end, num_layers)))
        self.architecture = architecture
        if type(loop_max_count) is not int or loop_max_count < 0:
            raise ValueError('loop_max_count must be a nonnegative integer')
        if type(loop_lora_rank) is not int or loop_lora_rank < 0:
            raise ValueError('loop_lora_rank must be a nonnegative integer')
        if not math.isfinite(loop_lora_alpha) or loop_lora_alpha <= 0:
            raise ValueError('loop_lora_alpha must be finite and positive')
        if architecture != 'loop' and (loop_max_count or loop_lora_rank):
            raise ValueError('Depth adapters require a loop architecture')
        self.loop_max_count = loop_max_count or loop_count
        if self.loop_max_count < loop_count:
            raise ValueError('Maximum depth cannot be smaller than default loop_count')
        self.loop_count = loop_count
        self.prelude_layers = prelude_layers
        self.recurrent_layers = recurrent_layers
        self.coda_layers = coda_layers
        self.loop_lora_rank = loop_lora_rank
        self.effective_depth = len(self.layer_order)
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self.apply(self._init_weights)
        if loop_lora_rank:
            self.loop_adapters = nn.ModuleList([
                nn.ModuleList([nn.ModuleDict({name: LoopLoRA(d_model, loop_lora_rank,
                    loop_lora_alpha, device, dtype) for name in ('q', 'v')})
                    for _ in range(recurrent_layers)]) for _ in range(self.loop_max_count)])

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Preserve the Base v1 initialization distributions with native layers."""
        if isinstance(module, nn.Linear):
            std = math.sqrt(2 / (module.in_features + module.out_features))
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)

    def execution_order(self, loop_count=None):
        count = self.loop_count if loop_count is None else loop_count
        if type(count) is not int or not 1 <= count <= self.loop_max_count:
            raise ValueError('Loop count outside configured range')
        if self.architecture != 'loop':
            return tuple((i, None) for i in range(len(self.layers)))
        p, r = self.prelude_layers, self.recurrent_layers
        return (tuple((i, None) for i in range(p))
                + tuple((p+j, k) for k in range(count) for j in range(r))
                + tuple((i, None) for i in range(p+r, len(self.layers))))

    def forward(self, token_ids: Tensor, output_positions=None,
                loop_count: int | None = None):
        """Return next-token logits with shape ``(..., sequence, vocab_size)``."""
        x = self.token_embeddings(token_ids)
        for index, iteration in self.execution_order(loop_count):
            adapters = (self.loop_adapters[iteration][index-self.prelude_layers]
                        if self.loop_lora_rank and iteration is not None else None)
            x = self.layers[index](x, adapters=adapters)
        if output_positions is not None:
            x = x[output_positions]
        return self.lm_head(self.ln_final(x))
