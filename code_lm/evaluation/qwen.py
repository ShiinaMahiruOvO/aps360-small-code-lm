"""Qwen2.5-Coder-0.5B-Instruct baseline: zero-shot, or SFT on the same tasks with natural-language prompts.

Each specification is translated into English (requirements), so Qwen gets the same information as our
models without the structured format. SFT mirrors configs/sft (6 epochs, effective batch 16, 3% warmup,
cosine to lr/10, weight decay 0.1, clipping 1.0, answer-token loss, best validation-loss checkpoint).
Sampling uses vLLM and never executes code; normalize_code extracts one deterministic candidate, and the
isolated sandbox grades it like every other model.

    python -m code_lm.evaluation.qwen --output runs/qwen-zero-shot
    python -m code_lm.evaluation.qwen --learning-rate 1e-4 --output runs/qwen-sft
"""
import argparse
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re

from code_lm.data.bst_dataset import validate
from code_lm.data.bst_tasks import ARGS, NODE_CODE
from code_lm.data.prepare import read_rows

MODEL, REVISION = 'Qwen/Qwen2.5-Coder-0.5B-Instruct', 'ea3f2471cf1b1f0db85067f1ef93848e38e88c25'
SYSTEM = ('You are Qwen, created by Alibaba Cloud. You are a helpful assistant. '
          'Implement the requested standalone Python function. Return code only, without explanations '
          'or example calls. TreeNode is already provided. Follow all stated return conventions.')
TESTS = ('test_seen', 'test_rl_pair', 'test_unseen_pair', 'test_holdout', 'test_transfer')  # field order and
# parameter names are not part of the English prompt, so the shuffled and renamed splits do not apply.


TRANSFER = {  # The test_transfer operations of code_lm.data.bst_dataset.
    'kth_smallest': 'Return the k-th smallest stored key, counting from k = 1. If k is less than 1 or larger than the number of keys, return -1.',
    'closest': 'Return the stored key with the smallest absolute difference from target; on a tie return the smaller key. For an empty tree return None.',
    'range_max': 'Return the largest stored key satisfying low <= key <= high. If there is none, return None.',
    'second_largest': 'Return the second largest stored key. If the tree has fewer than two keys, return None.',
    'count_nodes': 'Return the number of nodes in the tree.',
    'leaf_count': 'Return the number of leaf nodes (nodes with no children).',
    'height': 'Return the height of the tree: the number of edges on the longest path from the root to a leaf. For an empty tree return -1.',
    'lowest_common_ancestor': 'Keys a and b are both stored in the tree. Return the key of their lowest common ancestor (a node counts as its own ancestor).',
}


def requirements(spec):
    """Translate every specification field; no reference solutions or test examples."""
    f = spec['family']
    lines = ['The input is a valid binary search tree with distinct integer keys and at most 64 nodes.',
             'An empty tree is represented by None.']
    validate(spec)
    if f in TRANSFER:
        return '\n'.join(lines + [TRANSFER[f], 'Do not modify the tree.'])
    if f == 'search':
        lines += [f'Return a bool: True exactly when target is {"present" if spec["present"] else "absent"} in the tree, and False otherwise.']
    elif f == 'extreme':
        lines += [f'Return the {"maximum" if spec["maximum"] else "minimum"} stored key, not a node.',
                  f'For an empty tree return {spec["empty"]!r}.']
    elif f == 'neighbor':
        below = spec['direction'] == 'below'
        op = ('<' if below else '>') + ('=' if spec['inclusive'] else '')
        lines += [f'Among stored keys satisfying key {op} target, return the {"largest" if below else "smallest"} key.',
                  f'If there is no eligible key, return {spec["missing"]!r}. The target need not be stored.']
    elif f == 'search_depth':
        lines += [f'Return the number of edges from the root to target, plus {spec["base"]}.',
                  f'If target is absent, return {spec["missing"]!r} as the entire result.']
    elif f == 'search_path':
        lines += ['Return a list of the keys visited by an ordinary BST search for target.',
                  f'On a hit, {"include" if spec["include_target"] else "exclude"} the matching key.',
                  'On a miss, ' + ('keep the visited keys.' if spec['missing_path'] else 'return an empty list.'),
                  'List the keys ' + ('in reverse search order, ending at the root.' if spec['reverse'] else 'in search order, starting at the root.')]
    elif f in ('insert', 'insert_many'):
        lines += ['Use ordinary BST leaf insertion without rebuilding or rebalancing the tree.',
                  'Ignore duplicate keys. Preserve the identities of all existing nodes.']
        if f == 'insert':
            lines += ['Insert value.', 'Return ' + ('(final_root, added), where added is a bool indicating whether a new node was created.'
                                                  if spec['report_added'] else 'the final root.')]
        else:
            order = {'given':'their given order', 'reverse':'reverse input order', 'sorted':'ascending sorted order'}[spec['order']]
            lines += [f'Insert the integers in values one at a time in {order}.',
                      'values contains at most 8 integers. Do not mutate the input list.',
                      'Return ' + ('(final_root, added), where added is the number of newly created nodes.'
                                   if spec['report_added'] else 'the final root.')]
    else:
        lo = '<=' if spec['lower_closed'] else '<'
        hi = '<=' if spec['upper_closed'] else '<'
        lines += [f'low <= high. Select stored keys satisfying low {lo} key {hi} high.',
                  {'any':'Keep keys of either parity.', 'even':'Keep only even keys.', 'odd':'Keep only odd keys.'}[spec['parity']]]
        if f == 'range_count':
            lines += ['Return the number of selected keys; return 0 when there are none.']
        elif f == 'range_values':
            lines += [f'Return the selected keys as a list in {"descending" if spec["descending"] else "ascending"} order; return [] when there are none.']
        else:
            transform = {'identity':'unchanged values', 'square':'squares', 'abs':'absolute values', 'absolute':'absolute values'}[spec['transform']]
            lines += [f'After selection, sum the {transform} of the selected keys. Return 0 for an empty selection.']
    if f not in ('insert', 'insert_many'):
        lines += ['Do not modify the tree.']
    return '\n'.join(lines)


def natural_prompt(row):
    signature = next(line for line in row['prompt'].splitlines() if line.startswith(f'def {row["entry_point"]}('))
    return (requirements(row['spec']) + '\n\nProvided class:\n' + NODE_CODE
            + f'\nImplement this exact signature:\n{signature}\n'
            + 'Return the Python function and any helpers. Do not redefine TreeNode.')


def normalize_code(text, entry_point, family, args=None):
    """One deterministic candidate; no execution, oracle selection or logic edits.
    args: the parameter list, for operations outside ARGS."""
    args = args or ARGS[family]
    changes = []
    blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)', text, re.S)
    if blocks:
        code = next((b for b in blocks if re.search(rf'\bdef\s+{re.escape(entry_point)}\s*\(', b)), blocks[0])
        changes.append('extract_code_fence')
    else:
        code = text
    code = code.strip()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        lines = code.splitlines()
        starts = [i for i,l in enumerate(lines) if re.match(r'^(?:def |class |import |from |@)', l)]
        if starts and starts[0]:
            code = '\n'.join(lines[starts[0]:]);changes.append('remove_leading_prose')
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            lines = code.splitlines();line = (exc.lineno or len(lines))-1
            # Keep a complete requested function before trailing prose or an
            # unfinished subsequent module-level block. Never patch its body.
            tree = None
            cuts=[i for i,l in enumerate(lines) if 0<i<=line and l and not l[:1].isspace()]
            for cut in reversed(cuts):
                try:prefix=ast.parse('\n'.join(lines[:cut]))
                except SyntaxError:continue
                if any(isinstance(n,ast.FunctionDef) and n.name==entry_point for n in prefix.body):
                    tree=prefix;changes.append('remove_trailing_incomplete_module_block');break
            if tree is None:
                return code, changes + ['parse_error']
    kept = []
    allowed = (ast.Import,ast.ImportFrom,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Assign,ast.AnnAssign)
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=='TreeNode':
            changes.append('use_provided_TreeNode');continue
        if isinstance(node,ast.ImportFrom) and node.module=='__future__':
            changes.append('remove_future_import');continue
        if isinstance(node,allowed):kept.append(node)
        else:changes.append('remove_module_demo_or_prose')
    tree.body = kept
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.returns is not None:changes.append('remove_type_annotations');node.returns=None
        if isinstance(node,ast.arg) and node.annotation is not None:
            changes.append('remove_type_annotations');node.annotation=None
    if not any(isinstance(n,ast.FunctionDef) and n.name==entry_point for n in tree.body):
        argc=len(args.split(','))
        options=[n for n in tree.body if isinstance(n,ast.FunctionDef) and not n.name.startswith('_')
                 and len(n.args.posonlyargs)+len(n.args.args)==argc and not n.args.vararg and not n.args.kwarg]
        if len(options)==1:
            tree.body += ast.parse(f'{entry_point} = {options[0].name}').body
            changes.append('alias_unique_function_name')
        else:
            solution=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Solution'),None)
            methods=[] if solution is None else [n for n in solution.body if isinstance(n,ast.FunctionDef)
                and not n.name.startswith('_') and len(n.args.args)==argc+1 and n.args.args[0].arg=='self']
            matching=[n for n in methods if n.name==entry_point]
            if len(matching)==1 or len(methods)==1:
                name=(matching or methods)[0].name
                tree.body += ast.parse(f'def {entry_point}({args}):\n    return Solution().{name}({args})').body
                changes.append('unwrap_Solution_method')
            else:changes.append('missing_or_ambiguous_entry_point')
    return ast.unparse(tree)+'\n', sorted(set(changes))


def chat(tok, row):
    return tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': natural_prompt(row)}],
                                   tokenize=False, add_generation_prompt=True)


def encode(tok, row):
    prompt = tok(chat(tok, row), add_special_tokens=False)['input_ids']
    answer = tok(row['answer'].rstrip('\n') + '<|im_end|>', add_special_tokens=False)['input_ids']
    return prompt + answer, [-100] * len(prompt) + answer


def train(train_rows, val_rows, output, learning_rate, cache_dir=None, epochs=6, batch_size=8, grad_accum=2, seed=42):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir=cache_dir,
                                                 torch_dtype=torch.float32).cuda()
    train_data, val_data = [encode(tok, r) for r in train_rows], [encode(tok, r) for r in val_rows]
    decay = [p for p in model.parameters() if p.ndim >= 2]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                                   {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.}],
                                  lr=learning_rate, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train_data) / (batch_size * grad_accum))
    total, warmup = steps_per_epoch * epochs, int(steps_per_epoch * epochs * 0.03)
    schedule = lambda s: (s / max(warmup, 1) if s < warmup else
                          0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))

    def batches(data):
        for start in range(0, len(data), batch_size):
            chunk = data[start:start + batch_size]
            width = max(len(ids) for ids, _ in chunk)
            ids = torch.tensor([i + [tok.pad_token_id] * (width - len(i)) for i, _ in chunk]).cuda()
            labels = torch.tensor([l + [-100] * (width - len(l)) for _, l in chunk]).cuda()
            yield ids, labels, ids != tok.pad_token_id

    def loss_sum(ids, labels, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        return torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten(),
                                                 ignore_index=-100, reduction='sum')

    @torch.no_grad()
    def validate():
        model.eval()
        total_loss = sum(float(loss_sum(*b)) for b in batches(val_data))
        model.train()
        return total_loss / sum(sum(l != -100 for l in labels) for _, labels in val_data)
    best, log, step = validate(), [], 0
    log.append({'step': 0, 'validation_loss': best})
    model.save_pretrained(output); tok.save_pretrained(output)
    model.train()
    for epoch in range(epochs):
        order = list(range(len(train_data)))
        random.Random(seed + epoch).shuffle(order)
        for i in range(steps_per_epoch):
            update = [train_data[j] for j in order[i * batch_size * grad_accum:(i + 1) * batch_size * grad_accum]]
            tokens = sum(sum(l != -100 for l in labels) for _, labels in update)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate * schedule(step + 1)
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.
            for b in batches(update):
                loss = loss_sum(*b) / tokens
                loss.backward()
                train_loss += float(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 35 == 0 or i == steps_per_epoch - 1:
                val = validate()
                log.append({'step': step, 'train_loss': train_loss, 'validation_loss': val})
                print(log[-1], flush=True)
                if val < best:
                    best = val
                    model.save_pretrained(output); tok.save_pretrained(output)
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return {'best_validation_loss': best, 'steps': step, 'log': log}


def generate(model_path, rows, samples=8, temperature=1.0, seed=0, max_tokens=512, cache_dir=None):
    """Chat-format sampling; normalize_code makes one deterministic candidate per answer."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    local = Path(str(model_path)).is_dir()
    revision = None if local else REVISION
    tok = AutoTokenizer.from_pretrained(model_path, revision=revision, cache_dir=cache_dir)
    llm = LLM(model=str(model_path), revision=revision, tokenizer_revision=revision, download_dir=cache_dir,
              dtype='bfloat16', max_model_len=2048, gpu_memory_utilization=0.6, seed=seed)
    prompts = [chat(tok, r) for r in rows]
    outputs = llm.generate(prompts, SamplingParams(n=samples, temperature=temperature, top_p=1., max_tokens=max_tokens, seed=seed),
                           use_tqdm=False)
    result = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        for j, answer in enumerate(output.outputs):
            args = re.search(rf'def {row["entry_point"]}\((.*)\):', row['prompt']).group(1)
            code, changes = normalize_code(answer.text, row['entry_point'], row['family'], args)
            result.append({'id': f'{row["id"]}-sample{j}', 'source_id': row['id'], 'sample_index': j,
                           'completion': code, 'raw_completion': answer.text, 'normalization': changes,
                           'finish_reason': answer.finish_reason, 'temperature': temperature, 'seed': seed,
                           'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                           'model_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                           'model': MODEL, 'revision': REVISION, 'weights': str(model_path)})
    del llm
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=Path('data/bst'))
    parser.add_argument('--learning-rate', type=float, default=0., help='0 evaluates the released model zero-shot')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    weights = MODEL
    if args.learning_rate:
        weights = args.output / 'model'
        summary = train(list(read_rows(args.data / 'train.jsonl')), list(read_rows(args.data / 'validation.jsonl')),
                        weights, args.learning_rate, args.cache_dir)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    rows = [r for name in ('validation', *TESTS) for r in read_rows(args.data / f'{name}.jsonl')]
    by_split = {}
    for g in generate(weights, rows, cache_dir=args.cache_dir):
        by_split.setdefault(g['source_id'].split('-')[0], []).append(json.dumps(g) + '\n')
    for split, lines in by_split.items():
        (args.output / f'{split}.jsonl').write_text(''.join(lines))


if __name__ == '__main__':
    main()
