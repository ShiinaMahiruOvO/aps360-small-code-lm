"""Fixed-budget GRPO with injectable sampling, grading and update backends."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

from code_lm.data.prepare import read_rows, verify_dataset
from code_lm.evaluation.grader import build_jobs, summarize
from code_lm.training.grpo import advantages


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def validate(plan):
    for key in ('rounds','prompts_per_round','samples_per_prompt','max_tokens','update_epochs','validation_every','micro_batch'):
        if type(plan[key]) is not int or plan[key]<1:raise ValueError(f'Invalid {key}')
    if not 2<=plan['samples_per_prompt']<=8 or plan['samples_per_prompt']*plan['prompts_per_round']>128:
        raise ValueError('Require 2–8 samples per prompt and at most 128 sequences per round')
    if not 0<plan['clip_epsilon']<1 or not math.isfinite(plan['learning_rate']) or plan['learning_rate']<=0:
        raise ValueError('Invalid clipping or learning rate')
    if not math.isfinite(plan['kl_coefficient']) or plan['kl_coefficient']<0:raise ValueError('Invalid KL coefficient')
    if plan['precision'] not in ('fp32','bf16'):raise ValueError('Invalid precision')
    if plan.get('sampling_strategy','shuffle') not in ('shuffle','paired'):raise ValueError('Unknown sampling strategy')
    if plan.get('sampling_strategy')=='paired' and plan['prompts_per_round']%2:raise ValueError('Pairs need an even batch')
    if plan.get('difficulty_samples',0) not in (0,*range(2,9)) or not 1<=plan.get('validation_samples',1)<=8:
        raise ValueError('Use 2–8 difficulty samples (0 disables) and 1–8 validation samples')
    if type(plan.get('average_top',0)) is not int or plan.get('average_top',0)<0:raise ValueError('Invalid average_top')


def make_schedule(rows,pool,plan):
    """Shuffled epochs over rows. 'paired' follows each task with a least-used task that differs
    in exactly one condition, preferring rows and falling back to the full RL pool."""
    rng=random.Random(plan['seed']);schedule=[];exposure=Counter()
    neighbour=lambda a,b:a['family']==b['family'] and sum(a['spec'].get(k)!=b['spec'].get(k) for k in a['spec'].keys()|b['spec'].keys())==1
    while len(schedule)<plan['rounds']*plan['prompts_per_round']:
        epoch=sorted(rows,key=lambda r:(r['basic_contract'],r['spec_id']));rng.shuffle(epoch)
        if plan.get('sampling_strategy','shuffle')=='shuffle':schedule.extend(epoch);continue
        for r in epoch:
            choices=sorted((p for p in rows if neighbour(r,p)),key=lambda p:p['id']) or sorted((p for p in pool if neighbour(r,p)),key=lambda p:p['id'])
            if not choices:raise ValueError(f'No one-condition partner for {r["id"]}')
            rng.shuffle(choices);partner=min(choices,key=lambda p:exposure[p['id']])
            schedule+=[r,partner];exposure.update((r['id'],partner['id']))
    return schedule


def average_checkpoints(paths,output):
    import torch
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    model={k:(sum(s['model'][k].float() for s in states)/len(states)).to(v.dtype) for k,v in states[0]['model'].items()}
    torch.save({'model':model,'model_args':states[0]['model_args'],'stage':'rl_average','sources':[str(p) for p in paths]},output)


def run(checkpoint, data, output, plan, *, sample, grade_jobs, update_policy,
        resume=False, commit=lambda:None, backend='torch'):
    validate(plan)
    checkpoint,data,output=map(lambda p:Path(p).resolve(),(checkpoint,data,output))
    verify_dataset(data)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if output.exists() and not resume:raise FileExistsError('Use a new output directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    spec={'plan':plan,'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'backend':backend,
          'data_sha256':{p.name:digest(p) for p in sorted(data.iterdir()) if p.suffix in ('.json','.jsonl')}}
    frozen=output/'configuration.json'
    if frozen.exists() and json.loads(frozen.read_text())!=spec:raise ValueError('Resume configuration/data/checkpoint changed')
    write_json(frozen,spec)
    state_path=output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'completed_rounds':0,'checkpoint':str(checkpoint),'optimizer_steps':0,'updates':[],'curves':[]}
    if state.get('stage')=='completed':return json.loads((output/'comparison.json').read_text())
    def status(stage,**extra):
        if stage != 'failed': state.pop('error',None)
        state.update(stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
        write_json(state_path,state);commit();print(json.dumps({'stage':stage,'completed_rounds':state['completed_rounds'],**extra}),flush=True)
    suites=json.loads((data/'suites.json').read_text())
    rl=list(read_rows(data/'rl.jsonl'));validation=list(read_rows(data/'validation.jsonl'));tests=list(read_rows(data))
    if not rl or len({row['id'] for row in rl})!=len(rl):raise ValueError('RL prompts must be nonempty and unique')
    if {row['id'] for row in rl}&{row['id'] for row in validation+tests}:raise ValueError('RL and evaluation IDs overlap')
    cache_path=output/'grading-cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def generate(rows,path,label,samples=1,temperature=0.,seed=42):
        dest=output/'rollouts'/f'{label}.jsonl'
        if dest.exists():
            found=list(read_rows(dest))
            expected={(row['id'],j) for row in rows for j in range(samples)}
            if len(found)!=len(expected) or {(g['source_id'],g['sample_index']) for g in found}!=expected:
                raise ValueError('Cached rollout IDs differ')
            prompts={r['id']:hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows}
            policy_hash=digest(path)
            if any(g['checkpoint_sha256']!=policy_hash or g['temperature']!=temperature or g['seed']!=seed
                   or g['prompt_sha256']!=prompts[g['source_id']] for g in found):raise ValueError('Cached rollout policy/prompt differs')
            return found
        result=sample(rows,str(path),str(dest),samples=samples,temperature=temperature,seed=seed)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(''.join(json.dumps(row)+'\n' for row in result))
        commit();return result
    def grade(rows,generations,label):
        source={r['id']:r for r in rows}
        expanded=[{**source[g['source_id']],'id':g['id'],'source_id':g['source_id']} for g in generations]
        jobs,candidates=build_jobs(expanded,suites,generations)
        missing=[j for j in jobs if j['id'] not in cache]
        fresh=grade_jobs(missing) if missing else []
        merged={**cache,**{r['id']:r for r in fresh}}
        result=summarize(jobs,candidates,[merged[j['id']] for j in jobs])
        cache.update({r['id']:r for r in fresh});write_json(cache_path,cache)
        write_json(output/f'{label}-grade.json',result);commit();return result
    def evaluate(rows,path,label):
        status(label)
        return grade(rows,generate(rows,path,label),label)
    k=plan.get('validation_samples',1)
    def validate_at(path,label,round_index):
        status(label)
        result=grade(validation,generate(validation,path,label,k,1. if k>1 else 0.),label)
        state['curves'].append({'round':round_index,'correct':result['correct'],'tasks':result['tasks'],'checkpoint':str(path)})
        write_json(output/'curves.json',state['curves']);status('validated')
    def test(path,label):
        result=evaluate(tests,path,label);split={row['id']:row['split'] for row in tests}
        for name in dict.fromkeys(split.values()):
            selected=[r for r in result['outcomes'] if split[r['id']]==name]
            result[name]={'correct':sum(r['correct'] for r in selected),'tasks':len(selected)}
        return result
    try:
        if not state.get('initialized'):
            state['sft']=test(checkpoint,'sft-test')
            state['curves']=[];validate_at(checkpoint,'validation-0',0)
            status('initialized',initialized=True)
        if plan.get('difficulty_samples') and 'rl_pool' not in state:
            # Keep prompts the SFT policy sometimes but not always solves; constant groups carry no signal.
            gen=generate(rl,checkpoint,'difficulty',plan['difficulty_samples'],1.,plan['seed']+4)
            checked=grade(rl,gen,'difficulty')
            hits=Counter(r['source_id'] for r in checked['outcomes'] if r['correct'])
            state['rl_pool']=[r['id'] for r in rl if 0<hits[r['id']]<plan['difficulty_samples']]
            if not state['rl_pool']:raise ValueError('Difficulty filter removed every RL prompt')
            status('filtered',rl_pool_size=len(state['rl_pool']))
        completed=state['completed_rounds']
        if completed and completed%plan['validation_every']==0 and not any(c['round']==completed for c in state['curves']):
            validate_at(Path(state['checkpoint']),f'validation-{completed}',completed)
        pool=set(state.get('rl_pool',[r['id'] for r in rl]))
        schedule=make_schedule([r for r in rl if r['id'] in pool],rl,plan);n=plan['prompts_per_round']
        for i in range(state['completed_rounds'],plan['rounds']):
            start=time.monotonic();current=Path(state['checkpoint']);label=f'round-{i+1:04d}'
            batch=[{**r,'id':r['id']+f'-slot{j}'} for j,r in enumerate(schedule[i*n:(i+1)*n])]
            status(label+'-sampling')
            gen=generate(batch,current,label,plan['samples_per_prompt'],1.,plan['seed']+8+i)
            sampled=time.monotonic();status(label+'-grading')
            checked=grade(batch,gen,label);graded=time.monotonic()
            correct={r['id']:int(r['correct']) for r in checked['outcomes']}
            rewarded=[{**g,'reward':correct[g['id']]} for g in gen]
            update={'updated':False,'eligible_sequences':0,'optimizer_steps':0};target=current
            if advantages(rewarded):
                target=output/'checkpoints'/f'{label}.pt';status(label+'-updating')
                update=update_policy(rewarded,str(current),str(checkpoint),str(target))
                if not target.is_file():raise RuntimeError('Updater did not save its checkpoint')
            record={'round':i+1,'sample_correct':sum(correct.values()),'samples':len(gen),
                    'sampled_tokens':sum(len(g['answer_token_ids']) for g in gen),'update':update,
                    'sampling_seconds':sampled-start,'grading_seconds':graded-sampled,
                    'update_seconds':time.monotonic()-graded,'total_seconds':time.monotonic()-start}
            state['updates'].append(record)
            status('training',completed_rounds=i+1,checkpoint=str(target),
                   optimizer_steps=state['optimizer_steps']+update['optimizer_steps'])
            # Delete only superseded RL checkpoints after the new state is durable. A round without an
            # update validates the previous checkpoint, so keep every checkpoint a curve refers to.
            if current!=checkpoint and current!=target and str(current) not in {c['checkpoint'] for c in state['curves']}:
                current.unlink(missing_ok=True);current.with_suffix('.optimizer.pt').unlink(missing_ok=True)
            if (i+1)%plan['validation_every']==0:
                validate_at(target,f'validation-{i+1}',i+1)
        # Select on validation only: the best RL checkpoint (earliest on ties), or the
        # average of the top average_top checkpoints when it validates at least as well.
        ranked=sorted((c for c in state['curves'] if c['round']!='average' and c['round']>0),key=lambda c:(-c['correct'],c['round']))
        if not ranked:raise ValueError('No validated RL checkpoint; make rounds a multiple of validation_every')
        selected,selection=Path(ranked[0]['checkpoint']),{'rounds':[ranked[0]['round']],'validation_correct':ranked[0]['correct']}
        top=plan.get('average_top',0)
        if top>1 and len(ranked)>=top:
            averaged=output/'checkpoints'/'average.pt'
            if not averaged.exists():average_checkpoints([c['checkpoint'] for c in ranked[:top]],averaged);commit()
            if not any(c['round']=='average' for c in state['curves']):validate_at(averaged,'validation-average','average')
            score=next(c['correct'] for c in state['curves'] if c['round']=='average')
            if score>=ranked[0]['correct']:
                selected,selection=averaged,{'rounds':[c['round'] for c in ranked[:top]],'validation_correct':score}
        final=test(selected,'rl-test')
        comparison={'sft':state['sft'],'rl':final,'selection':{'rule':'validation pass rate',**selection,'checkpoint':str(selected)}}
        write_json(output/'comparison.json',comparison);write_json(output/'curves.json',state['curves'])
        status('completed');return comparison
    except Exception as exc:
        status('failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/rl.json'))
    p.add_argument('--data',type=Path,default=Path('data/bst'))
    p.add_argument('--tokenizer-dir',type=Path,default=Path('data/tokenizer'))
    p.add_argument('--output',type=Path,default=Path('runs/rl'))
    p.add_argument('--backend',choices=('torch','vllm'),default='torch')
    p.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda')
    p.add_argument('--grader-command',required=True,help='External isolated executor command with {input} and {output} file placeholders')
    p.add_argument('--resume',action='store_true');p.add_argument('--eager',action='store_true')
    a=p.parse_args();plan=json.loads(a.config.read_text());validate(plan)
    from code_lm.inference.rl_sampler import Sampler,TorchSampler
    from code_lm.evaluation.runner import IsolatedGrader
    from code_lm.training.rl_worker import TrainerState
    if a.backend=='vllm':
        from code_lm.inference.vllm_options import check_runtime
        if a.device!='cuda':p.error('vLLM requires --device cuda')
        check_runtime();sampler=Sampler(a.tokenizer_dir,plan['max_tokens'],a.eager)
    else:sampler=TorchSampler(a.tokenizer_dir,a.device,plan['max_tokens'])
    trainer=TrainerState(a.device,plan['learning_rate'],plan['update_epochs'],plan['clip_epsilon'],plan['kl_coefficient'])
    def update(rows,checkpoint,reference,output):
        return trainer.step(rows,checkpoint,reference,output,micro_batch=plan['micro_batch'],precision=plan['precision'])
    run(a.checkpoint,a.data,a.output,plan,sample=sampler.generate,
        grade_jobs=IsolatedGrader(a.grader_command),update_policy=update,
        resume=a.resume,backend=a.backend)


if __name__=='__main__':main()
