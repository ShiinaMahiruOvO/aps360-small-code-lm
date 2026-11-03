"""The BST dataset: every split, prompt and test suite, generated in one place.

Each of the 17 operations in bst_tasks.CONTRACTS has single-condition variants and two-condition
combinations. The combinations are split three ways: SFT (train), RL only (validation, rl,
test_rl_pair) and never trained (test_unseen_pair). Tests:

  test_seen            taught specifications under an unseen function name
  test_seen_shuffled   the same tasks and tests with the fields in a shuffled order
  test_seen_renamed    the same tasks and tests with renamed parameters
  test_rl_pair         the RL-only combinations
  test_unseen_pair     combinations never trained on
  test_holdout         every range specification with 3-4 changed conditions
  test_transfer        eight operations that never appear in training

    python -m code_lm.data.bst_dataset --output data/bst
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import types

from code_lm.data import bst_tasks as tasks

SEED, HOLDOUT_SEED, TRANSFER_SEED = 360930, 360923, 360960
TRAIN_NAMES = ('solve', 'tree_query', 'bst_operation', 'process_tree', 'compute', 'answer',
               'handle_tree', 'evaluate_bst', 'bst_task', 'run_query', 'tree_op', 'query_tree')
EVAL_NAMES = {'validation': 'query_bst', 'rl': 'apply_bst', 'test_seen': 'run_bst', 'test_unseen_pair': 'run_bst',
              'test_rl_pair': 'run_bst'}
SEED_OFFSET = {'train': 0, 'validation': 1000, 'rl': 2000, 'test_seen': 3000, 'test_unseen_pair': 4000, 'test_rl_pair': 5000}
FIELD_NAMES = {'present': 'true_when_present', 'empty': 'empty_result', 'missing': 'missing_result',
               'base': 'root_depth', 'missing_path': 'keep_path_on_miss', 'reverse': 'target_to_root'}
RENAMED = {'root': 'tree', 'target': 'key', 'value': 'key', 'values': 'keys', 'low': 'lo', 'high': 'hi'}
RANGE_CHOICES = {'lower_closed': [True, False], 'upper_closed': [True, False], 'parity': ['any', 'even', 'odd']}


def signature(spec):
    return json.dumps(spec, sort_keys=True, separators=(',', ':'))


def variants(base):
    """The base contract and its single changes, and every pair of changes to different fields."""
    choices = {}
    f = base['family']
    if f == 'extreme': choices['empty'] = [0, -1]
    if f in ('neighbor', 'search_depth'): choices['missing'] = [v for v in (None, 0, -1) if v != base['missing']]
    if f == 'search_path': choices = {'missing_path': [False], 'reverse': [True]}
    if f in ('insert', 'insert_many'): choices['report_added'] = [True]
    if f == 'insert_many': choices['order'] = ['reverse', 'sorted']
    if f.startswith('range_'): choices = {'lower_closed': [False], 'upper_closed': [False], 'parity': ['even', 'odd']}
    if f == 'range_values': choices['descending'] = [True]
    if f == 'range_sum': choices['transform'] = ['square', 'absolute']
    known = [deepcopy(base)]
    changes = [(k, v) for k, values in choices.items() for v in values]
    known += [{**base, k: v} for k, v in changes]
    combos = [{**base, k: v, l: w} for (k, v), (l, w) in itertools.combinations(changes, 2) if k != l]
    return known, combos


def holdout_specs():
    """Every range specification with three or more changes: all that variants() leaves out."""
    for family in ('range_count', 'range_values', 'range_sum'):
        choices = dict(RANGE_CHOICES)
        if family == 'range_values': choices['descending'] = [False, True]
        if family == 'range_sum': choices['transform'] = ['identity', 'square', 'absolute']
        known, combos = variants(tasks.CONTRACTS[family])
        covered = {signature(s) for s in known + combos}
        for values in itertools.product(*choices.values()):
            spec = {**tasks.CONTRACTS[family], **dict(zip(choices, values))}
            if signature(spec) not in covered:
                yield family, spec


def holdout_cases(spec):
    """The standard cases plus every endpoint pair on a mixed-sign, mixed-parity tree."""
    tree = [0, -3, 3, -4, -2, 2, 4, -1, 1]
    return tasks.cases(spec, 'validation', HOLDOUT_SEED) + [
        {'values': tree, 'args': [low, high], 'expected': tasks.expected(spec, tree, [low, high])}
        for low in range(-5, 6) for high in range(low, 6)]


# Transfer probe: (spec, parameters, trusted reference body). References only validate the tests.
TRANSFER = [
    ({'family': 'kth_smallest', 'missing': -1}, 'root, k',
     '    stack, node = [], root\n    while stack or node is not None:\n        while node is not None:\n'
     '            stack.append(node)\n            node = node.left\n        node = stack.pop()\n        k -= 1\n'
     '        if k == 0:\n            return node.val\n        node = node.right\n    return -1\n'),
    ({'family': 'closest', 'missing': None}, 'root, target',
     '    best = None\n    while root is not None:\n'
     '        if best is None or (abs(root.val - target), root.val) < (abs(best - target), best):\n'
     '            best = root.val\n        root = root.left if target < root.val else root.right\n    return best\n'),
    ({'family': 'range_max', 'lower_closed': True, 'upper_closed': True, 'missing': None}, 'root, low, high',
     '    best = None\n    while root is not None:\n        if root.val > high:\n            root = root.left\n'
     '        elif root.val < low:\n            root = root.right\n        else:\n            best = root.val\n'
     '            root = root.right\n    return best\n'),
    ({'family': 'second_largest', 'missing': None}, 'root',
     '    keys = []\n    def visit(node):\n        if node is not None:\n            visit(node.left)\n'
     '            keys.append(node.val)\n            visit(node.right)\n    visit(root)\n'
     '    return keys[-2] if len(keys) >= 2 else None\n'),
    ({'family': 'count_nodes'}, 'root',
     '    if root is None:\n        return 0\n    return 1 + {name}(root.left) + {name}(root.right)\n'),
    ({'family': 'leaf_count'}, 'root',
     '    if root is None:\n        return 0\n    if root.left is None and root.right is None:\n        return 1\n'
     '    return {name}(root.left) + {name}(root.right)\n'),
    ({'family': 'height', 'empty': -1}, 'root',
     '    if root is None:\n        return -1\n    return 1 + max({name}(root.left), {name}(root.right))\n'),
    ({'family': 'lowest_common_ancestor'}, 'root, a, b',
     '    while root is not None:\n        if a < root.val and b < root.val:\n            root = root.left\n'
     '        elif a > root.val and b > root.val:\n            root = root.right\n        else:\n            return root.val\n'
     '    return None\n'),
]
TRANSFER_ARGS = {spec['family']: args for spec, args, _ in TRANSFER}


def leaves(t):
    return 0 if t is None else 1 if t[1] is None and t[2] is None else leaves(t[1]) + leaves(t[2])


def height(t):
    return -1 if t is None else 1 + max(height(t[1]), height(t[2]))


def ancestor(t, a, b):
    while t is not None and (a < t[0] and b < t[0] or a > t[0] and b > t[0]):
        t = t[1] if a < t[0] else t[2]
    return t[0] if t is not None else None


def transfer_expected(spec, values, args):
    """Independent oracle over the sorted keys and the insertion-shape tuple."""
    f, keys, tree = spec['family'], sorted(set(values)), tasks.shape(values)
    if f == 'kth_smallest':
        return keys[args[0] - 1] if 1 <= args[0] <= len(keys) else spec['missing']
    if f == 'closest':
        return min(keys, key=lambda k: (abs(k - args[0]), k)) if keys else spec['missing']
    if f == 'range_max':
        inside = [k for k in keys if args[0] <= k <= args[1]]
        return inside[-1] if inside else spec['missing']
    if f == 'second_largest':
        return keys[-2] if len(keys) >= 2 else spec['missing']
    if f == 'count_nodes':
        return len(keys)
    if f == 'leaf_count':
        return leaves(tree)
    if f == 'height':
        return height(tree) if tree else spec['empty']
    return ancestor(tree, *args)


def transfer_cases(spec):
    """The trees of bst_tasks.cases with queries suited to each operation."""
    rng = random.Random(TRANSFER_SEED)
    trees = [[], [0], [-1], [8, 3, 12, 1, 5, 10, 15], [-5, -12, -1, -8, -3, 0],
             list(range(-32, 32)), list(range(31, -33, -1))]
    trees += [rng.sample(range(-150, 151), n) for n in (2, 5, 9, 16, 25, 40, 64)]
    rows = []
    for values in trees:
        lo, hi, f = min(values, default=-1), max(values, default=1), spec['family']
        if f == 'kth_smallest':
            queries = [[k] for k in dict.fromkeys([0, 1, 2, len(values), len(values) + 1])]
        elif f == 'closest':
            queries = [[v] for v in dict.fromkeys([lo - 5, hi + 5, 0, *(v + 1 for v in values[:3])])]
        elif f == 'range_max':
            queries = [[lo, hi], [lo, lo], [hi + 1, hi + 3], [-200, 200], [-7, 3], [0, 0]]
        elif f == 'lowest_common_ancestor':
            if not values:
                continue
            picks = sorted(values)
            queries = [[picks[0], picks[-1]], [picks[0], picks[0]], [picks[len(picks) // 2], picks[-1]]]
            queries += [sorted(rng.sample(values, 2)) for _ in range(3) if len(values) >= 2]
        else:
            queries = [[]]
        rows += [{'values': values, 'args': a, 'expected': transfer_expected(spec, values, a)} for a in queries]
    return rows


def arguments(spec):
    return TRANSFER_ARGS.get(spec['family']) or tasks.ARGS[spec['family']]


def validate(spec):
    """Accept exactly the specifications this dataset defines (for --spec inference and Qwen prompts)."""
    known = {signature(s) for base in tasks.CONTRACTS.values() for group in variants(base) for s in group}
    known |= {signature(s) for _, s in holdout_specs()} | {signature(s) for s, _, _ in TRANSFER}
    if not isinstance(spec, dict) or signature(spec) not in known:
        raise ValueError(f'Unknown BST specification: {spec}')


def render(spec, name, shuffle=None, rename=False):
    """The prompt. shuffle: a Random that reorders the fields (never leaving the usual order);
    rename: the RENAMED parameters."""
    fields = [f'operation = {spec["family"]!r}']
    fields += [f'{FIELD_NAMES.get(k, k)} = {v!r}' for k, v in sorted(spec.items()) if k != 'family']
    fields.append(f'modifies_tree = {spec["family"] in ("insert", "insert_many")}')
    usual = list(fields)
    while shuffle is not None and fields == usual:
        shuffle.shuffle(fields)
    args = arguments(spec)
    args = ', '.join(RENAMED[a] for a in args.split(', ')) if rename else args
    return ('Implement a Python function for this binary search tree task.\n'
            'Keys are distinct integers, the tree has at most 64 nodes, and None is an empty tree.\n\n'
            + '\n'.join(fields) + '\n\nProvided class:\n' + tasks.NODE_CODE
            + f'\nFunction signature:\ndef {name}({args}):\n'
            'Return only this Python function and nested helpers; do not redefine TreeNode.\n\nSolution:\n')


def suite(spec, name, cases):
    test = tasks.test_source(spec, 'validation', name, explicit_cases=cases)
    return hashlib.sha256(test.encode()).hexdigest(), {'tests': test, 'expected_tests': len(cases)}


def make_row(contract, spec, split, name, suites, strategy=0, index=0, suite_id=None, **variant):
    key = hashlib.sha256(signature(spec).encode()).hexdigest()[:16]
    if suite_id is None:
        suite_id, suites[suite_id] = suite(spec, name, tasks.cases(spec, 'validation', SEED + SEED_OFFSET[split]))
    return {'id': f'{split}-{contract}-{key}-{name}-{strategy}-{index}', 'group_id': f'{split}:{key}',
            'split': split, 'family': spec['family'], 'basic_contract': contract, 'spec_id': key, 'spec': spec,
            'entry_point': name, 'prompt': render(spec, name, **variant), 'answer': tasks.reference(spec, strategy, name),
            'suite_id': suite_id}


def passes(answer, tests):
    """Run a trusted reference against a suite in-process (never model output)."""
    module = types.ModuleType('solution')
    exec(tasks.NODE_CODE + answer, module.__dict__)
    sys.modules['solution'] = module
    namespace = {}
    try:
        exec(tests, namespace)
        for case in namespace['CASES']:
            namespace['test_contract'](case)
        return True
    except Exception:
        return False
    finally:
        sys.modules.pop('solution', None)


def build(output, copies=8, unseen_fraction=1/3):
    if output.exists():
        raise FileExistsError(output)
    rng = random.Random(SEED)
    taught, unseen, rl_only = [], [], []
    for contract, base in tasks.CONTRACTS.items():
        known, combos = variants(base)
        rng.shuffle(combos)
        cut = round(len(combos) * unseen_fraction)
        unseen += [(contract, s) for s in combos[:cut]]
        rl_only += [(contract, s) for s in combos[cut:2 * cut]]
        taught += [(contract, s) for s in known + combos[2 * cut:]]
    names = ('train', 'validation', 'rl', 'test_seen', 'test_unseen_pair', 'test_rl_pair',
             'test_seen_shuffled', 'test_seen_renamed', 'test_holdout')
    splits, suites = {name: [] for name in names}, {}
    for k, (contract, spec) in enumerate(taught):
        for i in range(copies):
            splits['train'].append(make_row(contract, spec, 'train', TRAIN_NAMES[(k + i) % len(TRAIN_NAMES)],
                                            suites, strategy=i % 2, index=i))
        for split in ('validation', 'rl', 'test_seen'):
            splits[split].append(make_row(contract, spec, split, EVAL_NAMES[split], suites))
        seen = splits['test_seen'][-1]
        splits['test_seen_shuffled'].append(make_row(contract, spec, 'test_seen_shuffled', 'run_bst', suites,
                                                     suite_id=seen['suite_id'], shuffle=random.Random(f'{SEED}-{seen["spec_id"]}')))
        splits['test_seen_renamed'].append(make_row(contract, spec, 'test_seen_renamed', 'run_bst', suites,
                                                    suite_id=seen['suite_id'], rename=True))
    for contract, spec in rl_only:
        for split in ('validation', 'rl', 'test_rl_pair'):
            splits[split].append(make_row(contract, spec, split, EVAL_NAMES[split], suites))
    for contract, spec in unseen:
        splits['test_unseen_pair'].append(make_row(contract, spec, 'test_unseen_pair', 'run_bst', suites))
    for family, spec in holdout_specs():
        suite_id, suites[suite_id] = suite(spec, 'run_bst', holdout_cases(spec))
        splits['test_holdout'].append(make_row(family, spec, 'test_holdout', 'run_bst', suites, suite_id=suite_id))
    transfer = []
    for spec, args, body in TRANSFER:
        suite_id, suites[suite_id] = suite(spec, 'run_bst', transfer_cases(spec))
        key = hashlib.sha256(signature(spec).encode()).hexdigest()[:16]
        transfer.append({'id': f'test_transfer-{spec["family"]}-{key}-run_bst-0-0', 'group_id': f'test_transfer:{key}',
                         'split': 'test_transfer', 'family': spec['family'], 'basic_contract': spec['family'],
                         'spec_id': key, 'spec': spec, 'entry_point': 'run_bst', 'prompt': render(spec, 'run_bst'),
                         'answer': 'def run_bst(' + args + '):\n' + body.replace('{name}', 'run_bst'),
                         'suite_id': suite_id})

    # Audits. Unseen specifications never reach training, and their prompts differ from training prompts.
    trained = {signature(r['spec']) for s in ('train', 'validation', 'rl') for r in splits[s]}
    assert not {signature(r['spec']) for r in splits['train']} & {signature(s) for _, s in rl_only}, 'RL-only pairs in SFT'
    renamed = lambda r: r['prompt'].replace(r['entry_point'], 'F')
    for rows in (splits['test_unseen_pair'], splits['test_holdout'], transfer):
        assert not trained & {signature(r['spec']) for r in rows}
        assert not {renamed(r) for r in splits['train']} & {renamed(r) for r in rows}
    assert len({(r['prompt'], r['answer']) for r in splits['train']}) == len(splits['train'])
    # Every reference passes its own tests, and no trained reference passes an unseen task's tests.
    everything = [r for rows in splits.values() for r in rows] + transfer
    assert all(passes(r['answer'], suites[r['suite_id']]['tests']) for r in everything)
    taught_refs = {signature(s): (s, contract) for contract, s in taught + rl_only}.values()
    for row in splits['test_unseen_pair'] + splits['test_holdout'] + transfer:
        arity = len(arguments(row['spec']).split(','))
        for spec, _ in taught_refs:
            if len(tasks.ARGS[spec['family']].split(',')) == arity:
                assert not passes(tasks.reference(spec, 0, 'run_bst'), suites[row['suite_id']]['tests']), (row['id'], spec)

    output.mkdir(parents=True)
    for split, rows in splits.items():
        random.Random(SEED).shuffle(rows)
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    splits['test_transfer'] = transfer
    (output / 'test_transfer.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in transfer))
    (output / 'suites.json').write_text(json.dumps(suites) + '\n')
    manifest = {'version': 'bst', 'seed': SEED, 'copies_per_spec': copies, 'unseen_fraction': unseen_fraction,
                'counts': {s: len(r) for s, r in splits.items()},
                'specs': {s: len({r['spec_id'] for r in rows}) for s, rows in splits.items()},
                'by_family': {s: dict(Counter(r['family'] for r in rows)) for s, rows in splits.items() if s.startswith('test')},
                'unseen_pair_specs': [spec for _, spec in unseen], 'rl_pair_specs': [spec for _, spec in rl_only],
                'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    m = build(p.parse_args().output)
    print(json.dumps({k: m[k] for k in ('counts', 'specs')}, indent=2))
