"""Optional inference integrations; importing the training package needs no vLLM."""


def register():
    """vLLM calls this plugin in its driver and worker processes."""
    import os
    # Adapter merging can initialize PyTorch's CPU thread pool before engine
    # creation. A forked worker can inherit locked thread-pool state.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "CodeLMForCausalLM", "code_lm.inference.vllm_model:CodeLMForCausalLM",
    )
