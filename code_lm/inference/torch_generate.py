"""Batched greedy generation for evaluation; never executes generated code."""

import math
import torch


@torch.inference_mode()
def generate_batch(model, prompts, *, max_new_tokens, eos_id, context_length, ignore_eos=False, temperature=0.0, seed=0):
    """Return completion IDs and stop reasons, without truncating any prompt.

    Right padding is safe for our causal model: valid positions cannot attend to
    later padding. Each row reads logits at its own last valid position, then
    overwrites its first padding slot with the next token.
    ignore_eos is only for fixed-length throughput comparisons; normal evaluation
    keeps its original EOS stopping behavior.
    """
    if max_new_tokens < 0 or not prompts or any(not row for row in prompts):
        raise ValueError("Require nonempty prompts and nonnegative max_new_tokens")
    if any(len(row) + max_new_tokens > context_length for row in prompts):
        raise ValueError("Prompt plus generation budget exceeds model context")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    lengths = torch.tensor([len(row) for row in prompts], device=device)
    initial_width = max(map(len, prompts))
    tokens = torch.full((len(prompts), initial_width + max_new_tokens),
                        eos_id, dtype=torch.long, device=device)
    for index, row in enumerate(prompts):
        tokens[index, :len(row)] = torch.tensor(row, device=device)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    indices = torch.arange(len(prompts), device=device)
    completions = [[] for _ in prompts]
    reasons = ["length"] * len(prompts)
    was_training = model.training
    model.eval()
    try:
        for step in range(max_new_tokens):
            logits = model(tokens[:, :initial_width + step])
            scores = logits[indices, lengths - 1].float()
            next_ids = (scores.argmax(dim=-1) if temperature == 0 else
                        torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1, generator=generator).squeeze(-1))
            active = ~finished
            tokens[indices[active], lengths[active]] = next_ids[active]
            lengths += active.long()
            for index, (token, is_active) in enumerate(zip(next_ids.tolist(), active.tolist())):
                if is_active:
                    if token == eos_id and not ignore_eos:
                        reasons[index] = "eos"
                    else:
                        completions[index].append(token)
            if not ignore_eos:
                finished |= next_ids == eos_id
            if bool(finished.all()):
                break
    finally:
        model.train(was_training)
    return [{"token_ids": ids, "new_tokens": len(ids), "finish_reason": reason}
            for ids, reason in zip(completions, reasons)]
