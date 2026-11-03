"""vLLM runtime checks, engine and sampling options for our exported model (Linux CUDA only)."""

import math
from pathlib import Path


def check_runtime():
    """Fail early with setup instructions for this project's CUDA backend."""
    import sys
    from importlib.metadata import PackageNotFoundError, version
    if sys.platform != 'linux':
        raise SystemExit('This vLLM backend requires Linux and an NVIDIA CUDA GPU. '
                         'For CPU inference use: python generate.py --backend torch --device cpu')
    try:
        installed = version('vllm')
    except PackageNotFoundError:
        raise SystemExit('vLLM is not installed. Follow the local vLLM setup in README.md.') from None
    if installed.split('+')[0] != '0.18.0':
        raise SystemExit(f'This adapter targets vLLM 0.18.0; found {installed}. Use the pinned README environment.')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU is available. Check the NVIDIA driver and CUDA-enabled PyTorch installation.')
    if torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit('This BF16 configuration requires an NVIDIA Ampere-or-newer GPU.')


def engine_options(model: Path, context_length: int, *, eager: bool = False) -> dict:
    """The single-GPU configuration measured on our 97M model with vLLM 0.18.

    The pool is deliberately small. Larger models/contexts need a new memory
    budget, rather than silently taking most of the GPU away from training.
    """
    return dict(model=str(model), skip_tokenizer_init=True, dtype='bfloat16', load_format='hf',
                tensor_parallel_size=1, enforce_eager=eager,
                compilation_config={'mode': 0, 'cudagraph_mode': 'NONE' if eager else 'FULL_DECODE_ONLY',
                                    'cudagraph_capture_sizes': [] if eager else [1, 2, 4, 8]},
                max_model_len=context_length, max_num_seqs=8,
                max_num_batched_tokens=max(4096, context_length),
                kv_cache_memory_bytes=512 * 1024**2, seed=0,
                enable_prefix_caching=False, enable_chunked_prefill=False)


def sampling_options(eos_id: int, max_new_tokens: int = 128, temperature: float = 0.0,
                     top_p: float = 1.0, samples: int = 1, seed: int = 0) -> dict:
    if max_new_tokens <= 0 or not 1 <= samples <= 8:
        raise ValueError('Require positive output length and 1–8 samples per prompt')
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError('Require finite temperature >= 0 and 0 < top_p <= 1')
    if temperature == 0 and samples != 1:
        raise ValueError('Greedy generation produces one sample; use temperature > 0 for multiple samples')
    # skip_tokenizer_init disables automatic EOS inference in this vLLM version.
    return dict(n=samples, temperature=temperature, top_p=top_p, seed=seed,
                max_tokens=max_new_tokens, detokenize=False, stop_token_ids=[eos_id])
