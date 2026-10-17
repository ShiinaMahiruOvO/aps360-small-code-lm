"""Compare native Qwen interfaces on the frozen BST tasks using isolated grading."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from code_lm.data.bst_tasks import ARGS, NODE_CODE
from code_lm.data.bst_spec import render_spec
MODELS = {
    'instruct': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', 'ea3f2471cf1b1f0db85067f1ef93848e38e88c25'),
    'base': ('Qwen/Qwen2.5-Coder-0.5B', '8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301'),
}
SYSTEM = ('You are Qwen, created by Alibaba Cloud. You are a helpful assistant. '
          'Implement the requested standalone Python function. Return code only, without explanations '
          'or example calls. TreeNode is already provided. Follow all stated return conventions.')


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def requirements(spec):
    """Translate every specification field; no reference solutions or test examples."""
    render_spec(spec)  # Validate the exact schema before translating it.
    f = spec['family']
    lines = ['The input is a valid binary search tree with distinct integer keys and at most 64 nodes.',
             'An empty tree is represented by None.']
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
    return (requirements(row['spec']) + '\n\nProvided class:\n' + NODE_CODE
            + f'\nImplement this exact signature:\ndef {row["entry_point"]}({ARGS[row["family"]]}):\n'
            + 'Return the Python function and any helpers. Do not redefine TreeNode.')


def completion_prefix(row):
    doc = '\n'.join('    ' + line for line in requirements(row['spec']).splitlines())
    return f'def {row["entry_point"]}({ARGS[row["family"]]}):\n    """\n{doc}\n    """\n'


def normalize_code(text, entry_point, family):
    """One deterministic candidate; no execution, oracle selection or logic edits."""
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
        argc=len(ARGS[family].split(','))
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
                name=(matching or methods)[0].name;args=ARGS[family]
                tree.body += ast.parse(f'def {entry_point}({args}):\n    return Solution().{name}({args})').body
                changes.append('unwrap_Solution_method')
            else:changes.append('missing_or_ambiguous_entry_point')
    return ast.unparse(tree)+'\n', sorted(set(changes))


def worker(root, kind, cache_dir=None):
    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model,revision=MODELS[kind]
    tok=AutoTokenizer.from_pretrained(model,revision=revision,cache_dir=cache_dir)
    requests=json.loads((root/(kind+'-requests.json')).read_text())
    for row in requests:
        row['model_prompt']=(tok.apply_chat_template([{'role':'system','content':SYSTEM},
            {'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
            if kind=='instruct' else row['prompt'])
    (root/(kind+'-rendered-prompts.json')).write_text(json.dumps(requests,indent=2))
    start=time.monotonic()
    llm=LLM(model=model,revision=revision,tokenizer_revision=revision,download_dir=cache_dir,
            dtype='bfloat16',max_model_len=8192,max_num_seqs=64,gpu_memory_utilization=.7,
            enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=42)
    init_seconds=time.monotonic()-start;start=time.monotonic()
    outputs=llm.generate([r['model_prompt'] for r in requests],SamplingParams(temperature=0.,top_p=1.,n=1,
                         max_tokens=2048,seed=42),use_tqdm=False)
    rows=[]
    for request,output in zip(requests,outputs,strict=True):
        answer=output.outputs[0]
        rows.append({'id':request['id'],'protocol':request['protocol'],
            'completion':request.get('completion_prefix','')+answer.text,'raw_completion':answer.text,
            'prompt_sha256':digest(request['prompt']),'model_prompt_sha256':digest(request['model_prompt']),
            'finish_reason':answer.finish_reason,'output_tokens':len(answer.token_ids),
            'prompt_tokens':len(output.prompt_token_ids),'model':model,'revision':revision})
    (root/(kind+'-generations.json')).write_text(json.dumps(rows,indent=2))
    (root/(kind+'-metrics.json')).write_text(json.dumps({'model':model,'revision':revision,
        'initialization_seconds':init_seconds,'generation_seconds':time.monotonic()-start,
        'finish_reasons':dict(Counter(r['finish_reason'] for r in rows)),'answers':len(rows)},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=tuple(MODELS), required=True)
    parser.add_argument('--data', type=Path, default=Path('data/bst_covered'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    from code_lm.data.prepare import read_rows
    rows = list(read_rows(args.data))
    requests = []
    for row in rows:
        protocols = (('instruct-structured', row['prompt']),
                     ('instruct-natural', natural_prompt(row))) if args.kind == 'instruct' else (
                     ('base-completion', NODE_CODE+'\n'+completion_prefix(row)),)
        for protocol, prompt in protocols:
            requests.append({'id': row['id'], 'prompt': prompt, 'protocol': protocol,
                             'completion_prefix': completion_prefix(row) if args.kind == 'base' else ''})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/(args.kind+'-requests.json')).write_text(json.dumps(requests, indent=2))
    worker(args.output, args.kind, str(args.cache_dir) if args.cache_dir else None)


if __name__ == '__main__':
    main()
