"""Single-GPU native vLLM adapter, initially tested against vLLM 0.18.0.

Training owns the mathematical reference in model.py. This module owns packed
token layout and the engine boundary. vLLM owns scheduling and paged KV caches.
No Llama weight conversion, Q/K permutation, or model-generated code execution.
"""

from collections.abc import Iterable

import torch
from torch import nn

from code_lm.model import RMSNorm, RotaryPositionalEmbedding, SwiGLU


class PackedAttention(nn.Module):
    """Project packed tokens, apply our adjacent-pair RoPE, then call the engine."""

    def __init__(self, config, engine: nn.Module):
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = width // self.num_heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.output_proj = nn.Linear(width, width, bias=False)
        self.rope = RotaryPositionalEmbedding(
            config.rope_theta, self.head_dim, config.max_position_embeddings,
        )
        self.engine = engine

    def forward(self, x, positions):
        # N is the total scheduled tokens, NOT a batch or a sequence length.
        # positions can be [0,1,2, 0,1] for two unrelated prompts, or [3,2]
        # when decoding one new token from each of those requests.
        shape = (-1, self.num_heads, self.head_dim)
        query = self.q_proj(x).view(shape)
        key = self.k_proj(x).view(shape)
        value = self.v_proj(x)
        query = self.rope(query, positions[:, None]).flatten(1)
        key = self.rope(key, positions[:, None]).flatten(1)
        # Do NOT apply a global triangular mask to the packed token axis.
        # Engine metadata supplies request boundaries, slots and causal masking.
        output = self.engine(query.contiguous(), key.contiguous(), value.contiguous())
        return self.output_proj(output)


class PackedBlock(nn.Module):
    def __init__(self, config, engine):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = PackedAttention(config, engine)
        self.ln2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x, positions):
        x = x + self.attn(self.ln1(x), positions)
        return x + self.ffn(self.ln2(x))


def load_named_weights(module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Require every parameter exactly once, with an exact shape match.

    Buffers such as RoPE tables and engine scales are not checkpoint parameters.
    Do not silently ignore unknown keys or initialize missing weights randomly.
    Load weights only between completed generation calls. In-place copying preserves CUDA graph addresses.
    """
    params = dict(module.named_parameters())
    loaded = set()
    with torch.no_grad():
        for name, value in weights:
            if name not in params or name in loaded:
                raise ValueError(f"Unexpected or duplicate parameter: {name}")
            if params[name].shape != value.shape:
                raise ValueError(f"Shape mismatch for {name}: {value.shape} != {params[name].shape}")
            params[name].copy_(value)
            loaded.add(name)
    if missing := params.keys() - loaded:
        raise ValueError(f"Missing parameters: {sorted(missing)}")
    return loaded


class CodeLMForCausalLM(nn.Module):
    """Native vLLM entry point. No TP, PP, quantization or LoRA. Fixed loops are expanded by the exporter."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        # Deferred import lets CPU tests exercise the math without installing
        # Linux/CUDA-specific vLLM dependencies on the developer's Mac.
        from vllm.model_executor.layers.attention import Attention

        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("This adapter currently supports one GPU, TP=PP=1")
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise ValueError("Quantization and LoRA are not supported by this adapter")
        config = vllm_config.model_config.hf_config
        if (config.num_key_value_heads != config.num_attention_heads
                or config.tie_word_embeddings or config.hidden_act != "silu"
                or config.attention_bias or config.mlp_bias
                or getattr(config, "code_lm_format", None) != 1):
            raise ValueError("Expected our version-1 full-MHA, bias-free, untied checkpoint")
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        blocks = []
        for index in range(config.num_hidden_layers):
            # This is an execution/cache identity, not a learned parameter.
            # Future fixed loops need a distinct identity per unrolled invocation
            # even when their projection/FFN parameters are shared.
            cache_name = f"{prefix + '.' if prefix else ''}layers.{index}.attn.engine"
            engine = Attention(
                num_heads=config.num_attention_heads,
                head_size=config.hidden_size // config.num_attention_heads,
                scale=(config.hidden_size // config.num_attention_heads) ** -0.5,
                num_kv_heads=config.num_key_value_heads,
                cache_config=vllm_config.cache_config,
                prefix=cache_name,
            )
            blocks.append(PackedBlock(config, engine))
        self.layers = nn.ModuleList(blocks)
        self.ln_final = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def embed_input_ids(self, input_ids):
        return self.token_embeddings(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("Pipeline parallel intermediate tensors are not supported")
        x = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            x = layer(x, positions)
        # vLLM selects the required token positions before projecting to vocab.
        return self.ln_final(x)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        return load_named_weights(self, weights)
