"""Canonical BST contracts for SFT and explicit-spec inference (not an NL parser)."""
import argparse
from collections import Counter
import hashlib
import json
import keyword
from pathlib import Path

from code_lm.data.contracts import CONTRACTS
from code_lm.data.bst_tasks import ARGS, NODE_CODE


def render_spec(spec, entry_point='solve'):
    if not isinstance(spec,dict) or spec.get('family') not in ARGS:
        raise ValueError('Unknown BST operation family')
    if not isinstance(entry_point,str) or not entry_point.isidentifier() or keyword.iskeyword(entry_point):
        raise ValueError('Invalid function name')
    family=spec['family']
    template=next(s for s in CONTRACTS.values() if s['family']==family)
    if set(spec)!=set(template): raise ValueError('Specification fields must exactly match the operation schema')
    for key,value in spec.items():
        if key=='family':continue
        if key in ('missing','empty'):
            if value is not None and type(value) is not int:raise ValueError(f'{key} must be null or an integer')
        elif key=='base':
            if type(value) is not int or value not in (0,1):raise ValueError('base must be 0 or 1')
        elif type(template[key]) is bool:
            if type(value) is not bool:raise ValueError(f'{key} must be boolean')
        else:
            choices={'direction':{'above','below'},'order':{'given','reverse','sorted'},
                     'parity':{'any','even','odd'},'transform':{'identity','square','abs','absolute'}}
            if not isinstance(value,str) or value not in choices[key]:raise ValueError(f'Invalid {key}')
    fields={'TASK':'BST','OP':family.upper(),'KEYS':'DISTINCT_INTEGERS','MAX_NODES':64,'EMPTY_TREE':'None'}
    if family=='search':fields.update(RETURN='BOOL',TRUE_WHEN='PRESENT' if spec['present'] else 'ABSENT')
    elif family=='extreme':fields.update(SELECT='MAXIMUM' if spec['maximum'] else 'MINIMUM',EMPTY_RESULT=spec['empty'])
    elif family=='neighbor':fields.update(DIRECTION=spec['direction'].upper(),EQUALITY=spec['inclusive'],MISSING_RESULT=spec['missing'],SELECT='CLOSEST_ELIGIBLE_KEY')
    elif family=='search_depth':fields.update(ROOT_INDEX=spec['base'],MISSING_RESULT=spec['missing'],RETURN='TARGET_DEPTH')
    elif family=='search_path':fields.update(INCLUDE_TARGET=spec['include_target'],ON_MISS='KEEP_VISITED' if spec['missing_path'] else 'EMPTY_LIST',ORDER='TARGET_TO_ROOT' if spec['reverse'] else 'ROOT_TO_TARGET',RETURN='KEY_LIST')
    elif family in ('insert','insert_many'):
        fields.update(DUPLICATES='IGNORE',PRESERVE_EXISTING_NODES=True,RETURN='ROOT_AND_ADDED' if spec['report_added'] else 'ROOT')
        if family=='insert_many':fields.update(INPUT_ORDER=spec['order'].upper(),MAX_VALUES=8,MUTATE_VALUES=False)
    else:
        fields.update(LOWER_BOUND='CLOSED' if spec['lower_closed'] else 'OPEN',UPPER_BOUND='CLOSED' if spec['upper_closed'] else 'OPEN',BOUNDS='LOW_LE_HIGH',PARITY=spec['parity'].upper())
        if family=='range_values':fields.update(ORDER='DESCENDING' if spec['descending'] else 'ASCENDING',EMPTY_RESULT='[]',RETURN='KEY_LIST')
        elif family=='range_sum':fields.update(TRANSFORM=spec['transform'].upper(),EMPTY_RESULT=0,RETURN='SUM')
        else:fields.update(EMPTY_RESULT=0,RETURN='COUNT')
    fields['MUTATE_TREE']=family in ('insert','insert_many')
    def text(v):
        if v is None:return 'None'
        if type(v) is bool:return 'TRUE' if v else 'FALSE'
        return str(v)
    contract='\n'.join(f'{k}={text(v)}' for k,v in fields.items())
    return ('Implement the following structured BST specification.\n\n'+contract+'\n\nProvided class:\n'+NODE_CODE+
            f'\nFunction signature:\ndef {entry_point}({ARGS[family]}):\n'
            'Return only this Python function and nested helpers; do not redefine TreeNode.\n\nSolution:\n')


