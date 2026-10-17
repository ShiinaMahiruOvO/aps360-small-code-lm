"""Generate the final 552-example BST SFT dataset and frozen evaluations."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random

from code_lm.data.contracts import CONTRACTS
from code_lm.data.bst_spec import render_spec
from code_lm.data import bst_tasks as tasks


def signature(spec):
    return json.dumps(spec,sort_keys=True,separators=(',',':'))


def variants(base):
    choices={}
    f=base['family']
    if f=='extreme':choices['empty']=[0,-1]
    if f in ('neighbor','search_depth'):choices['missing']=[v for v in (None,0,-1) if v!=base['missing']]
    if f=='search_path':choices={'missing_path':[False],'reverse':[True]}
    if f in ('insert','insert_many'):choices['report_added']=[True]
    if f=='insert_many':choices['order']=['reverse','sorted']
    if f.startswith('range_'):choices={'lower_closed':[False],'upper_closed':[False],'parity':['even','odd']}
    if f=='range_values':choices['descending']=[True]
    if f=='range_sum':choices['transform']=['square','absolute']
    known=[deepcopy(base)]
    changes=[(k,v) for k,values in choices.items() for v in values]
    for k,v in changes:known.append({**base,k:v})
    combos=[]
    for (k,v),(l,w) in itertools.combinations(changes,2):
        if k!=l:combos.append({**base,k:v,l:w})
    return known,combos


def build(output,seed=360921):
    if output.exists():raise FileExistsError(output)
    splits={k:[] for k in ('train','validation','rl','test_known','test_combinations')};suites={}
    seed_offset={'train':0,'validation':1000,'rl':2000,'test_known':3000,'test_combinations':4000}
    def row(contract,spec,split,name,strategy=0,index=0):
        key=hashlib.sha256(signature(spec).encode()).hexdigest()[:16]
        prompt=render_spec(spec,name)
        test=tasks.test_source(spec,'validation',name,seed+seed_offset[split])
        sid=hashlib.sha256(test.encode()).hexdigest()
        suites[sid]={'tests':test,'expected_tests':len(tasks.cases(spec,'validation',seed+seed_offset[split]))}
        return {'id':f'balanced-{split}-{contract}-{key}-{name}-{strategy}-{index}',
                'group_id':f'{split}:{key}', 'split':split,'family':spec['family'],'basic_contract':contract,
                'spec_id':key,'spec':spec,'entry_point':name,'prompt':prompt,
                'answer':tasks.reference(spec,strategy,name),'suite_id':sid,'input_format':'bst_spec_v1'}
    for contract,base in CONTRACTS.items():
        known,combos=variants(base)
        # Equal original-operation quotas; rotate specs before reference strategies/names.
        taught=known+combos
        quota=max(32,2*len(taught))
        for i in range(quota):
            spec=taught[i%len(taught)];cycle=i//len(taught)
            name_index=(i%len(taught)+cycle)%4
            splits['train'].append(row(contract,spec,'train',('solve','tree_query','bst_operation','process_tree')[name_index],cycle%2,i))
        for spec in known:
            for split,name in [('validation','query_bst'),('rl','apply_bst'),('test_known','run_bst')]:
                splits[split].append(row(contract,spec,split,name))
        for spec in combos:
            for split,name in [('validation','query_bst'),('rl','apply_bst')]:
                splits[split].append(row(contract,spec,split,name))
            splits['test_combinations'].append(row(contract,spec,'test_combinations','run_bst'))
    known_keys={x['spec_id'] for x in splits['train']}
    assert all(x['spec_id'] in known_keys for x in splits['validation']+splits['test_known'])
    assert all(x['spec_id'] in known_keys for x in splits['test_combinations'])
    counts=Counter(x['spec_id'] for x in splits['train'])
    assert min(counts.values())>=2
    assert {x['spec_id'] for x in splits['test_known']+splits['test_combinations']}==known_keys
    prompt_sets=[{x['prompt'] for x in rows} for rows in splits.values()]
    assert all(not a&b for a,b in itertools.combinations(prompt_sets,2))
    rng=random.Random(seed)
    for split, rows in splits.items():
        rng.shuffle(rows)
    output.mkdir(parents=True)
    for split,rows in splits.items():(output/f'{split}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (output/'suites.json').write_text(json.dumps(suites)+'\n')
    manifest={'seed':seed,'counts':{k:len(v) for k,v in splits.items()},
              'train_per_contract':dict(Counter(x['basic_contract'] for x in splits['train'])),
              'unique_train_prompts':len({x['prompt'] for x in splits['train']}),'known_specs':len(known_keys),
              }
    manifest.update(version='covered-v2',test_spec_coverage=1.0,minimum_examples_per_spec=min(counts.values()),
                    deduplication='Weighted quotas: 32 per original operation, increased to 40 for range_sum to teach every spec twice.',
                    split_policy='All 87 test specifications are taught in SFT; function names and random test seeds are split-specific. These are development evaluations, not unseen-combination generalization.')
    manifest['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=360921)
    a=p.parse_args();print(json.dumps(build(a.output,a.seed),indent=2))
