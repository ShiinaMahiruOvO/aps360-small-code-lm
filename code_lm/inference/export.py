"""Package our checkpoint for the CUSTOM adapter, without changing its weights.

Usage: python -m code_lm.inference.export --checkpoint ... --output ... --eos-id 256
The Llama *configuration schema* describes dimensions; architectures selects our
own model implementation. These are NOT weights for LlamaForCausalLM.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from code_lm.model import TransformerLM, model_args_from_config, inference_weights


def expand_weights(state, layer_order):
    """Serving copies of shared layers get separate KV-cache identities.

    Training retains its original shared parameters. Copies are refreshed on
    every policy update; inference does not introduce trainable parameters.
    """
    for name, value in state.items():
        if not name.startswith('layers.'):
            yield name, value
    for target, source in enumerate(layer_order):
        prefix = f'layers.{source}.'
        for name, value in state.items():
            if name.startswith(prefix):
                yield f'layers.{target}.' + name[len(prefix):], value


def materialize_loop(state, args, loop_count=None):
    """Merge depth-specific LoRA into separate serving layers at a fixed depth."""
    with torch.device('meta'):
        model = TransformerLM(**args)
    model.load_state_dict(state, strict=True, assign=True)
    order = model.execution_order(loop_count)
    expanded = {k: v for k, v in state.items()
                if not k.startswith(('layers.', 'loop_adapters.'))}
    for target, (source, iteration) in enumerate(order):
        prefix = f'layers.{source}.'
        for name, value in state.items():
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if model.loop_lora_rank and iteration is not None and suffix in ('attn.q_proj.weight', 'attn.v_proj.weight'):
                projection = 'q' if suffix == 'attn.q_proj.weight' else 'v'
                adapter = model.loop_adapters[iteration][source-model.prelude_layers][projection]
                value = value + (adapter.b @ adapter.a) * adapter.scale
            expanded[f'layers.{target}.{suffix}'] = value
    plain_args = {**args, 'architecture': 'standard', 'num_layers': len(order),
                  'prelude_layers': 0, 'recurrent_layers': 0, 'coda_layers': 0,
                  'loop_count': 1, 'loop_max_count': 0, 'loop_lora_rank': 0}
    return expanded, plain_args


def export_checkpoint(checkpoint: Path, output: Path, eos_id: int):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved.get("config") or {}
    source = saved.get("model_args") or metadata.get("model_args") or metadata
    args = model_args_from_config(source)
    state, args = inference_weights(saved['model'], args)
    if not 0 <= eos_id < args["vocab_size"]:
        raise ValueError("EOS must be in the unchanged vocabulary")
    # Meta allocation checks names/shapes without constructing another full model.
    with torch.device("meta"):
        reference = TransformerLM(**args)
    reference.load_state_dict(state, strict=True, assign=True)
    config = {
        "model_type": "llama", "architectures": ["CodeLMForCausalLM"], "code_lm_format": 1,
        "vocab_size": args["vocab_size"], "hidden_size": args["d_model"],
        "intermediate_size": args["d_ff"], "num_hidden_layers": reference.effective_depth,
        "code_lm_source_layer_order": list(reference.layer_order),
        "code_lm_source_model_args": args,
        "num_attention_heads": args["num_heads"], "num_key_value_heads": args["num_heads"],
        "max_position_embeddings": args["context_length"], "rope_theta": args["rope_theta"],
        "rms_norm_eps": 1e-5, "hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
        "tie_word_embeddings": False, "bos_token_id": None, "eos_token_id": eos_id,
        "pad_token_id": eos_id, "torch_dtype": "float32",
    }
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # vLLM's HF/bin loader accepts a plain named tensor dictionary. Optimizer state
    # never enters the serving artifact. No renaming or permutation is performed.
    weights = (materialize_loop(state, args)[0] if args.get('loop_lora_rank')
               else dict(expand_weights(state, reference.layer_order)))
    torch.save(weights, output / "pytorch_model.bin")
    with checkpoint.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"checkpoint_sha256": source_hash, "model_args": args, "eos_id": eos_id,
                "architecture": "CodeLMForCausalLM", "weight_transform": "depth-specific LoRA merge and layer expansion" if args.get('loop_lora_rank') else ("execution-order layer expansion" if args["architecture"] == "loop" else "none"),
                "tokenizer": "External tokenizer; pass prompt_token_ids, no automatic BOS or chat template."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eos-id", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output, args.eos_id), indent=2))
