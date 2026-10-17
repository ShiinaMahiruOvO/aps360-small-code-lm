"""BST reference templates and independent list/set/tuple correctness oracles.

Recovered from the original project generators; no candidate code executes here.
"""
import ast
import random

NODE_CODE = '''class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
'''


def shape(values):
    """Independent insertion-order oracle using partitions, without TreeNode."""
    unique = list(dict.fromkeys(values))
    if not unique:
        return None
    root, *rest = unique
    return (root, shape([v for v in rest if v < root]), shape([v for v in rest if v > root]))


FIXTURE = '''
def build(values):
    root = None
    for value in values:
        if root is None:
            root = TreeNode(value)
            continue
        node = root
        while True:
            if value == node.val:
                break
            field = 'left' if value < node.val else 'right'
            child = getattr(node, field)
            if child is None:
                setattr(node, field, TreeNode(value))
                break
            node = child
    return root

def snapshot(root):
    seen = set()
    nodes = {}
    def visit(node):
        if node is None:
            return None
        assert type(node) is TreeNode
        assert type(node.val) is int
        assert id(node) not in seen, 'cycle or shared child'
        seen.add(id(node))
        assert len(seen) <= 65, 'too many nodes'
        nodes[id(node)] = node.val
        return (node.val, visit(node.left), visit(node.right))
    return visit(root), nodes
'''

ARGS = {
    'search': 'root, target', 'search_path': 'root, target', 'search_depth': 'root, target',
    'extreme': 'root', 'insert': 'root, value', 'insert_many': 'root, values',
    'range_sum': 'root, low, high', 'range_count': 'root, low, high',
    'range_values': 'root, low, high', 'neighbor': 'root, target',
}


def reference(spec, strategy=0, name='solve'):
    """Two bounded implementation strategies; specification remains the ground truth."""
    f = spec['family']
    header = f'def {name}({ARGS[f]}):\n'
    if f == 'search':
        yes, no = repr(spec['present']), repr(not spec['present'])
        body = (f'    if root is None:\n        return {no}\n    if root.val == target:\n        return {yes}\n'
                f'    return {name}(root.left if target < root.val else root.right, target)\n' if strategy else
                f'    while root is not None:\n        if root.val == target:\n            return {yes}\n'
                f'        root = root.left if target < root.val else root.right\n    return {no}\n')
    elif f == 'search_path':
        body = '    path = []\n    node = root\n    while node is not None:\n'
        body += '        if node.val == target:\n'
        if spec['include_target']:
            body += '            path.append(node.val)\n'
        body += '            break\n        path.append(node.val)\n'
        if strategy:
            body += '        if target < node.val:\n            node = node.left\n        else:\n            node = node.right\n'
        else:
            body += '        node = node.left if target < node.val else node.right\n'
        if not spec['missing_path']:
            body += '    if node is None:\n        return []\n'
        body += '    return path[::-1]\n' if spec['reverse'] else '    return path\n'
    elif f == 'search_depth':
        if strategy:
            body = (f'    def walk(node, depth):\n        if node is None:\n            return {spec["missing"]!r}\n'
                    '        if node.val == target:\n            return depth\n'
                    '        return walk(node.left if target < node.val else node.right, depth + 1)\n'
                    f'    return walk(root, {spec["base"]})\n')
        else:
            body = (f'    depth = {spec["base"]}\n    while root is not None:\n'
                    '        if target == root.val:\n            return depth\n'
                    '        root = root.left if target < root.val else root.right\n        depth += 1\n'
                    f'    return {spec["missing"]!r}\n')
    elif f == 'extreme':
        field = 'right' if spec['maximum'] else 'left'
        body = f'    if root is None:\n        return {spec["empty"]!r}\n'
        body += (f'    if root.{field} is None:\n        return root.val\n    return {name}(root.{field})\n' if strategy else
                 f'    while root.{field} is not None:\n        root = root.{field}\n    return root.val\n')
    elif f in ('insert', 'insert_many'):
        if strategy:
            body = ('    def add(node, key):\n        if node is None:\n            return TreeNode(key), True\n'
                    '        if key == node.val:\n            return node, False\n'
                    '        if key < node.val:\n            node.left, added = add(node.left, key)\n'
                    '        else:\n            node.right, added = add(node.right, key)\n        return node, added\n')
        else:
            body = ('    def add(node, key):\n        if node is None:\n            return TreeNode(key), True\n'
                    '        current = node\n        while True:\n            if key == current.val:\n                return node, False\n'
                    '            if key < current.val:\n                if current.left is None:\n'
                    '                    current.left = TreeNode(key)\n                    return node, True\n'
                    '                current = current.left\n            else:\n                if current.right is None:\n'
                    '                    current.right = TreeNode(key)\n                    return node, True\n'
                    '                current = current.right\n')
        if f == 'insert':
            body += '    root, added = add(root, value)\n'
        else:
            seq = {'given': 'values', 'reverse': 'values[::-1]', 'sorted': 'sorted(values)'}[spec['order']]
            body += f'    added = 0\n    for key in {seq}:\n        root, changed = add(root, key)\n        added += int(changed)\n'
        body += '    return (root, added)\n' if spec['report_added'] else '    return root\n'
    elif f.startswith('range_'):
        lc = '<=' if spec['lower_closed'] else '<'
        uc = '<=' if spec['upper_closed'] else '<'
        cond = f'low {lc} node.val {uc} high'
        if spec['parity'] != 'any':
            cond += f' and node.val % 2 == {0 if spec["parity"] == "even" else 1}'
        if strategy:
            body = ('    def visit(node):\n        if node is None:\n            return []\n'
                    f'        middle = [node.val] if {cond} else []\n'
                    '        left = visit(node.left) if node.val >= low else []\n'
                    '        right = visit(node.right) if node.val <= high else []\n'
                    '        return left + middle + right\n    selected = visit(root)\n')
        else:
            body = ('    selected = []\n    stack = []\n    node = root\n    while stack or node is not None:\n'
                    '        while node is not None:\n            stack.append(node)\n            node = node.left\n'
                    f'        node = stack.pop()\n        if {cond}:\n            selected.append(node.val)\n'
                    '        node = node.right\n')
        if f == 'range_count':
            body += '    return len(selected)\n'
        elif f == 'range_values':
            body += '    return selected[::-1]\n' if spec['descending'] else '    return selected\n'
        else:
            expr = {'identity': 'key', 'square': 'key * key', 'absolute': 'abs(key)', 'abs': 'abs(key)'}[spec['transform']]
            body += f'    return sum({expr} for key in selected)\n'
    else:
        below = spec['direction'] == 'below'
        cmp = ('<=' if spec['inclusive'] else '<') if below else ('>=' if spec['inclusive'] else '>')
        good, bad = ('right', 'left') if below else ('left', 'right')
        if strategy:
            body = (f'    def find(node, best):\n        if node is None:\n            return best\n'
                    f'        if node.val {cmp} target:\n            return find(node.{good}, node.val)\n'
                    f'        return find(node.{bad}, best)\n    return find(root, {spec["missing"]!r})\n')
        else:
            body = (f'    best = {spec["missing"]!r}\n    while root is not None:\n'
                    f'        if root.val {cmp} target:\n            best = root.val\n            root = root.{good}\n'
                    f'        else:\n            root = root.{bad}\n    return best\n')
    code = header + body
    ast.parse(code)
    return code


def expected(spec, values, args):
    """Independent list/set/tuple oracle, not an invocation of a code template."""
    f = spec['family']
    keys = sorted(set(values))
    if f == 'search':
        return (args[0] in keys) == spec['present']
    if f in ('search_path', 'search_depth'):
        tree, target, path = shape(values), args[0], []
        while tree is not None:
            path.append(tree[0])
            if target == tree[0]:
                break
            tree = tree[1] if target < tree[0] else tree[2]
        if f == 'search_depth':
            return len(path) - 1 + spec['base'] if tree else spec['missing']
        if tree is None and not spec['missing_path']:
            path = []
        elif tree is not None and not spec['include_target']:
            path = path[:-1]
        return path[::-1] if spec['reverse'] else path
    if f == 'extreme':
        return (keys[-1] if spec['maximum'] else keys[0]) if keys else spec['empty']
    if f in ('insert', 'insert_many'):
        seq = [args[0]] if f == 'insert' else args[0][:]
        if f == 'insert_many':
            seq = seq[::-1] if spec['order'] == 'reverse' else sorted(seq) if spec['order'] == 'sorted' else seq
        added = len(set(seq) - set(values))
        return {'tree': shape(values + seq), 'added': bool(added) if f == 'insert' else added}
    if f == 'neighbor':
        target = args[0]
        eligible = [k for k in keys if (k < target if spec['direction'] == 'below' else k > target)
                    or (spec['inclusive'] and k == target)]
        return (eligible[-1] if spec['direction'] == 'below' else eligible[0]) if eligible else spec['missing']
    low, high = args
    eligible = [k for k in keys if (k > low or (spec['lower_closed'] and k == low))
                and (k < high or (spec['upper_closed'] and k == high))
                and (spec['parity'] == 'any' or k % 2 == (0 if spec['parity'] == 'even' else 1))]
    if f == 'range_count':
        return len(eligible)
    if f == 'range_values':
        return eligible[::-1] if spec['descending'] else eligible
    return sum(k if spec['transform'] == 'identity' else k*k if spec['transform'] == 'square' else abs(k)
               for k in eligible)


def cases(spec, split, seed=42):
    rng = random.Random(seed + {'train': 3602101, 'validation': 3602202}[split])
    trees = [[], [0], [-1], [8, 3, 12, 1, 5, 10, 15], [-5, -12, -1, -8, -3, 0],
             list(range(-32, 32)), list(range(31, -33, -1))]
    trees += [rng.sample(range(-150, 151), n) for n in (2, 5, 9, 16, 25, 40, 64)]
    rows = []
    for values in trees:
        lo, hi = min(values, default=-1), max(values, default=1)
        f = spec['family']
        if f.startswith('range_'):
            queries = [[lo, hi], [lo, lo], [hi, hi], [hi+1, hi+3], [-200, 200], [-7, 3], [0, 0]]
        elif f == 'extreme':
            queries = [[]]
        elif f == 'insert_many':
            queries = [[v] for v in ([], [0, 0], [hi+1, lo-1, hi+1], values[:3],
                                    rng.sample(range(-160, 161), 8))]
        else:
            queries = [[v] for v in dict.fromkeys([lo-1, hi+1, 0, *values[:2], *values[-2:]])]
        for args in queries:
            rows.append({'values': values, 'args': args, 'expected': expected(spec, values, args)})
    return rows


def test_source(spec, split, name='solve', seed=42, *, explicit_cases=None):
    f = spec['family']
    # Batch insert has at most 64 + 8 nodes; reject cycles and accidental growth.
    fixture = FIXTURE.replace('<= 65', '<= 72')
    code = (f'from solution import TreeNode, {name}\nimport copy\nimport pytest\n' + fixture +
            f'\nCASES = {cases(spec, split, seed) if explicit_cases is None else explicit_cases!r}\n'
            '@pytest.mark.parametrize("case", CASES)\ndef test_contract(case):\n'
            '    root = build(case["values"])\n    before, old_nodes = snapshot(root)\n'
            '    args = copy.deepcopy(case["args"])\n'
            f'    result = {name}(root, *args)\n    assert args == case["args"], "input arguments mutated"\n')
    if f in ('insert', 'insert_many'):
        if spec['report_added']:
            typ = 'bool' if f == 'insert' else 'int'
            code += ('    assert type(result) is tuple and len(result) == 2\n    result, added = result\n'
                     f'    assert type(added) is {typ}\n    assert added == case["expected"]["added"]\n')
        code += ('    after, new_nodes = snapshot(result)\n    assert after == case["expected"]["tree"]\n'
                 '    assert all(new_nodes.get(k) == v for k, v in old_nodes.items())\n'
                 '    if root is not None:\n        assert result is root\n')
    else:
        code += ('    assert type(result) is type(case["expected"])\n    assert result == case["expected"]\n'
                 '    if isinstance(result, list):\n        assert all(type(v) is int for v in result)\n'
                 '    assert snapshot(root) == (before, old_nodes)\n')
    return code
