"""Small clipped group-relative update for execution-reward experiments.

Caller supplies on-policy token sequences and binary rewards from isolated tests.
Only answer tokens (including EOS) receive loss; constant-reward groups are skipped.
"""
from collections import defaultdict
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F


def advantages(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['source_id']].append(i)
    result={}
    for ids in groups.values():
        rewards=torch.tensor([rows[i]['reward'] for i in ids],dtype=torch.float32)
        if not all(float(x) in (0,1) for x in rewards):raise ValueError('Expected binary execution rewards')
        std=rewards.std(unbiased=False)
        if std>0:
            for i,a in zip(ids,((rewards-rewards.mean())/std).tolist()):result[i]=a
    return result


def log_probs(model,rows):
    device=next(model.parameters()).device
    seqs=[r['prompt_token_ids']+r['answer_token_ids'] for r in rows]
    if any(not r['prompt_token_ids'] or not r['answer_token_ids'] for r in rows):raise ValueError('Empty sequence')
    width=max(map(len,seqs));tokens=torch.zeros((len(rows),width),dtype=torch.long,device=device)
    for i,seq in enumerate(seqs):tokens[i,:len(seq)]=torch.tensor(seq,device=device)
    lengths=[len(r['answer_token_ids']) for r in rows]
    positions=[(i,j) for i,r in enumerate(rows)
               for j in range(len(r['prompt_token_ids'])-1,len(seqs[i])-1)]
    batch_index,token_index=torch.tensor(positions,device=device).unbind(1)
    logits=model(tokens[:,:-1],output_positions=(batch_index,token_index)).float()
    selected=-F.cross_entropy(logits,tokens[batch_index,token_index+1],reduction='none')
    return list(selected.split(lengths))


def clipped_loss(current,old,reference,advantage,epsilon=.2,beta=.02):
    delta=current-old;ratio=delta.exp()
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    diff=reference-current
    kl=diff.exp()-diff-1
    return (-surrogate+beta*kl).mean(), ((ratio-1).abs()>epsilon).float().mean().detach(), kl.mean().detach()


def update(model,reference,optimizer,rows,epochs=2,micro_batch=8,precision='fp32',epsilon=.2,beta=.02):
    if epochs<1 or micro_batch<1 or precision not in ('fp32','bf16'):
        raise ValueError('Invalid update settings')
    if not 0 < epsilon < 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError('Invalid clipping or KL coefficient')
    device=next(model.parameters()).device
    if precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires a supported CUDA device')
    def autocast():
        return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
    adv=advantages(rows);eligible=[{**rows[i],'advantage':a} for i,a in adv.items()]
    if not eligible:return {'updated':False,'eligible_sequences':0,'optimizer_steps':0}
    model.eval();reference.eval()
    for p in reference.parameters():p.requires_grad_(False)
    old=[];ref=[]
    with torch.no_grad():
        for start in range(0,len(eligible),micro_batch):
            batch=eligible[start:start+micro_batch]
            missing=[row for row in batch if 'behavior_log_probs' not in row]
            with autocast():
                measured=iter(log_probs(model,missing) if missing else [])
            for row in batch:
                if 'behavior_log_probs' in row:
                    values=row['behavior_log_probs']
                    if len(values)!=len(row['answer_token_ids']) or not all(math.isfinite(v) for v in values):
                        raise ValueError('Invalid sampler log probabilities')
                    behavior=torch.tensor(row['behavior_log_probs'],device=device,dtype=torch.float32)
                    old.append(behavior)
                else:old.append(next(measured).detach().clone())
            with autocast():
                ref.extend([x.detach().clone() for x in log_probs(reference,batch)])
    logs=[]
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True);metrics=torch.zeros(3,device=device)
        for start in range(0,len(eligible),micro_batch):
            batch=eligible[start:start+micro_batch]
            with autocast():current=log_probs(model,batch)
            lengths=torch.tensor([len(r['answer_token_ids']) for r in batch],device=device)
            weights=(1/lengths.float()).repeat_interleave(lengths)/len(eligible)
            advantage=torch.tensor([r['advantage'] for r in batch],device=device).repeat_interleave(lengths)
            lp=torch.cat(current);ratio=(lp-torch.cat(old[start:start+len(batch)])).exp()
            diff=torch.cat(ref[start:start+len(batch)])-lp;kl=diff.exp()-diff-1
            losses=-torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)+beta*kl
            total=(losses*weights).sum()
            metrics+=torch.stack((total.detach(),(((ratio-1).abs()>epsilon).float()*weights).sum().detach(),
                                  (kl*weights).sum().detach()))
            total.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();values=metrics.tolist()
        logs.append({'epoch':epoch,'loss':values[0],'clip_fraction':values[1],'sampled_kl':values[2],'grad_norm':float(norm)})
    return {'updated':True,'eligible_sequences':len(eligible),'optimizer_steps':epochs,'epochs':logs}
