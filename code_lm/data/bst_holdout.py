"""Freeze all previously untaught combinations of the existing range conditions.

This is a prospective compositional holdout, not a new-operation benchmark.
Only repository-authored reference templates execute during the overlap audit.
"""
import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from code_lm.data import bst_tasks as tasks
from code_lm.data.bst_dataset import signature
from code_lm.data.bst_spec import render_spec
from code_lm.data.contracts import CONTRACTS


def build(source, output, seed=360923):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    old = [json.loads(line) for split in ('train', 'validation', 'rl', 'test_known', 'test_combinations')
           for line in (source / (split + '.jsonl')).read_text().splitlines()]
    taught = {signature(row['spec']) for row in old}
    train = [json.loads(line) for line in (source / 'train.jsonl').read_text().splitlines()]
    # Reconstruct trusted references and demand exact agreement with saved targets.
    references = {}
    for row in train:
        choices = [tasks.reference(row['spec'], strategy, row['entry_point']) for strategy in (0, 1)]
        if row['answer'] not in choices:
            raise ValueError('Unexpected training reference; do not execute arbitrary dataset text')
        code = tasks.reference(row['spec'], choices.index(row['answer']), 'run_bst')
        references[code] = row['family']
    ns = {}
    exec(tasks.NODE_CODE + tasks.FIXTURE, ns)
    old_functions = []
    for code, family in references.items():
        local = dict(ns)
        exec(code, local)
        old_functions.append((code, local['run_bst'], family))
    rows, suites, audits = [], {}, []
    for family in ('range_count', 'range_values', 'range_sum'):
        choices = {'lower_closed': [True, False], 'upper_closed': [True, False],
                   'parity': ['any', 'even', 'odd']}
        if family == 'range_values': choices['descending'] = [False, True]
        if family == 'range_sum': choices['transform'] = ['identity', 'square', 'absolute']
        for values in itertools.product(*choices.values()):
            spec = {**CONTRACTS[family], **dict(zip(choices, values))}
            if signature(spec) in taught: continue
            case_rows = tasks.cases(spec, 'validation', seed)
            # Include every pair of endpoints on a mixed-sign, mixed-parity tree.
            tree = [0, -3, 3, -4, -2, 2, 4, -1, 1]
            for low in range(-5, 6):
                for high in range(low, 6):
                    args = [low, high]
                    case_rows.append({'values': tree, 'args': args,
                                      'expected': tasks.expected(spec, tree, args)})
            key = hashlib.sha256(signature(spec).encode()).hexdigest()[:16]
            answer = tasks.reference(spec, 0, 'run_bst')
            assert answer not in references
            def passes(fn, case):
                root = ns['build'](case['values'])
                before = ns['snapshot'](root)
                try:
                    value = fn(root, *case['args'])
                    return (type(value) is type(case['expected']) and value == case['expected']
                            and ns['snapshot'](root) == before)
                except Exception:
                    return False
            for strategy in (0, 1):
                local = dict(ns); exec(tasks.reference(spec, strategy, 'run_bst'), local)
                assert all(passes(local['run_bst'], c) for c in case_rows), (spec, strategy)
            rejected = 0
            for code, fn, old_family in old_functions:
                # All other signatures differ from (root, low, high).
                if len(tasks.ARGS[old_family].split(',')) != 3:
                    rejected += 1; continue
                witness = next((i for i, c in enumerate(case_rows) if not passes(fn, c)), None)
                if witness is None:
                    raise ValueError(f'An old complete solution passes holdout {spec}')
                rejected += 1
            tests = tasks.test_source(spec, 'validation', 'run_bst', seed, explicit_cases=case_rows)
            sid = hashlib.sha256(tests.encode()).hexdigest()
            suites[sid] = {'tests': tests, 'expected_tests': len(case_rows)}
            rows.append({'id': 'holdout-' + family + '-' + key, 'split': 'holdout',
                         'family': family, 'basic_contract': family, 'spec_id': key, 'spec': spec,
                         'entry_point': 'run_bst', 'prompt': render_spec(spec, 'run_bst'),
                         'answer': answer, 'suite_id': sid, 'input_format': 'bst_spec_v1'})
            audits.append({'id': rows[-1]['id'], 'old_references_rejected': rejected,
                           'execution_cases': len(case_rows), 'reference_strategies_verified': 2})
    assert len(rows) == 27
    output.mkdir(parents=True)
    (output / 'test.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (output / 'suites.json').write_text(json.dumps(suites) + '\n')
    manifest = {'version': 'compositional-holdout-v1', 'seed': seed, 'tasks': len(rows),
                'by_family': dict(Counter(r['family'] for r in rows)),
                'selection': 'Every untaught combination in the existing finite range specification grammar.',
                'limitations': 'Three range operations only; held-out tasks combine 3–4 changes and are harder than taught tasks. Templates and atomic conditions are shared. No claim of unknown pretraining provenance.',
                'policy': 'No training, checkpoint selection, prompt tuning, or budget changes based on these results.',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(source.glob('*.jsonl'))},
                'training_reference_count': len(references), 'exact_solution_overlap': 0,
                'specification_overlap': 0, 'audit': audits,
                'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in output.iterdir()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/bst_covered'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m = build(a.source, a.output)
    print(json.dumps({k: m[k] for k in ('tasks', 'by_family', 'training_reference_count', 'exact_solution_overlap')}))
