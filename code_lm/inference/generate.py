"""Generate with an exported custom model on a Linux CUDA machine, no Modal.

The tokenizer files are the ORIGINAL vocab.json/merges.txt, not a Llama tokenizer.
"""

import argparse
import json
import math
from pathlib import Path

from code_lm.tokenizer import Tokenizer


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompt')
    inputs.add_argument('--prompts-file', type=Path, help='JSON array of prompt strings; no automatic chat template')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eager', action='store_true', help='Disable CUDA Graphs for debugging')
    args = parser.parse_args()
    check_runtime()
    config = json.loads((args.model / 'config.json').read_text())
    if config.get('architectures') != ['CodeLMForCausalLM']:
        raise ValueError('Expected an export for our custom model')
    tokenizer = Tokenizer.from_files(args.tokenizer_dir / 'vocab.json',
                                    args.tokenizer_dir / 'merges.txt', ['<|endoftext|>'])
    if set(tokenizer.vocab) != set(range(config['vocab_size'])):
        raise ValueError('Tokenizer vocabulary size/IDs differ from the model')
    if tokenizer.token_to_id[b'<|endoftext|>'] != config['eos_token_id']:
        raise ValueError('EOS ID differs from model configuration')
    sampling_kwargs = sampling_options(config['eos_token_id'], args.max_new_tokens,
                                       args.temperature, args.top_p, args.samples, args.seed)
    prompts = json.loads(args.prompts_file.read_text()) if args.prompts_file else [args.prompt]
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError('Expected a nonempty JSON array of prompt strings')
    prompt_ids = [tokenizer.encode(prompt) for prompt in prompts]
    if any(not ids or len(ids) + args.max_new_tokens > config['max_position_embeddings'] for ids in prompt_ids):
        raise ValueError('Every prompt must be nonempty and fit context together with its output budget')

    # These imports only happen when generation is actually requested.
    from vllm import LLM, SamplingParams
    from code_lm.inference import register

    register()
    llm = LLM(**engine_options(args.model, config['max_position_embeddings'], eager=args.eager))
    sampling = SamplingParams(**sampling_kwargs)
    outputs = llm.generate([{'prompt_token_ids': ids} for ids in prompt_ids], sampling, use_tqdm=False)
    for prompt_index, request in enumerate(outputs):
        for sample_index, completion in enumerate(request.outputs):
            print(json.dumps({'prompt_index': prompt_index, 'sample_index': sample_index,
                              'text': tokenizer.decode(completion.token_ids),
                              'token_ids': list(completion.token_ids),
                              'finish_reason': completion.finish_reason,
                              'stop_reason': completion.stop_reason}, ensure_ascii=False))


if __name__ == '__main__':
    main()
