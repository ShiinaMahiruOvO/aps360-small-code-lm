"""Resident GRPO state, with a durable optimizer sidecar for exact recovery."""
import hashlib
import json
from pathlib import Path
import time
import uuid

import torch

from code_lm.training.grpo import update
from code_lm.training.sft import load_model


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def cpu_state(value):
    if isinstance(value,torch.Tensor):return value.detach().to(device='cpu',copy=True)
    if isinstance(value,dict):return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_state(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_state(v) for v in value)
    return value


def atomic_save(value,path):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


class TrainerState:
    def __init__(self,device='cuda',learning_rate=1e-6,epochs=2,epsilon=.2,beta=.02):
        self.learning_rate,self.epochs,self.epsilon,self.beta=learning_rate,epochs,epsilon,beta
        self.device=torch.device(device)
        self.engine_id=str(uuid.uuid4())
        self.current=None
        self.reference_path=None
        self.model=None
        self.reference=None
        self.optimizer=None

    def load(self,path,reference_path):
        path,reference_path=Path(path),Path(reference_path)
        if self.current==path and self.reference_path==reference_path:
            return False
        self.model=None;self.optimizer=None
        self.model,self.args,saved=load_model(path,self.device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.learning_rate,weight_decay=0)
        optimizer_state=None
        if 'grpo_optimizer' in saved:
            optimizer_state=saved['grpo_optimizer']
        elif 'grpo_optimizer_file' in saved:
            sidecar=path.parent/saved['grpo_optimizer_file']
            if sidecar.parent!=path.parent or digest(sidecar)!=saved['grpo_optimizer_sha256']:
                raise ValueError('Invalid optimizer sidecar')
            optimizer_state=torch.load(sidecar,map_location='cpu',weights_only=True)
        elif saved.get('stage')=='grpo':
            raise ValueError('Cannot resume GRPO without optimizer state')
        if saved.get('stage') == 'grpo' and saved.get('reference_checkpoint') != str(reference_path):
            raise ValueError('Reference checkpoint differs from resumed RL run')
        if optimizer_state is not None:self.optimizer.load_state_dict(optimizer_state)
        for group in self.optimizer.param_groups:group['lr']=self.learning_rate
        del saved
        if self.reference_path!=reference_path:
            self.reference=None
            self.reference,_,saved=load_model(reference_path,self.device)
            del saved
            self.reference.requires_grad_(False)
        self.reference_path=reference_path
        self.current=path
        return True

    def step(self,rows,path,reference_path,out,*,micro_batch=8,precision='fp32',
             prune_checkpoint=False,commit=lambda:None):
        start=time.monotonic()
        path,reference_path,out=(Path(p).resolve() for p in (path,reference_path,out))
        if not 1<=len(rows)<=128:raise ValueError('Update exceeds sequence budget')
        request_hash=hashlib.sha256(json.dumps({'rows':rows,'checkpoint':str(path),'reference':str(reference_path),
            'micro_batch':micro_batch,'precision':precision,'learning_rate':self.learning_rate,
            'epochs':self.epochs,'epsilon':self.epsilon,'beta':self.beta},sort_keys=True).encode()).hexdigest()
        metadata=out.with_suffix('.json')
        if out.exists():
            prior=json.loads(metadata.read_text())
            if prior['request_sha256']!=request_hash or prior['output_sha256']!=digest(out):
                raise ValueError('Existing output belongs to a different request')
            return prior
        source_hash=digest(path)
        if any(r['checkpoint_sha256']!=source_hash or r['temperature']!=1. for r in rows):
            raise ValueError('Stale/off-policy rollouts')
        if prune_checkpoint and (path.parent!=out.parent or path==out or not path.name.startswith('round-')):
            raise ValueError('Only superseded RL checkpoints in the same directory may be pruned')
        try:
            cold=self.load(path,reference_path)
            if self.device.type=='cuda':torch.cuda.synchronize()
            loaded=time.monotonic()
            result=update(self.model,self.reference,self.optimizer,rows,micro_batch=micro_batch,precision=precision,
                          epochs=self.epochs,epsilon=self.epsilon,beta=self.beta)
            if self.device.type=='cuda':torch.cuda.synchronize()
            trained=time.monotonic()
            out.parent.mkdir(parents=True,exist_ok=True)
            sidecar=out.with_suffix('.optimizer.pt')
            atomic_save(cpu_state(self.optimizer.state_dict()),sidecar)
            # Sampler reads only model weights; recovery also restores Adam moments.
            atomic_save({'model':cpu_state(self.model.state_dict()),'model_args':self.args,
                'stage':'grpo','reference_checkpoint':str(reference_path),
                'grpo_optimizer_file':sidecar.name,'grpo_optimizer_sha256':digest(sidecar)},out)
            result['optimizer_total_steps']=sorted({int(v['step']) for v in self.optimizer.state.values() if 'step' in v})
            result['learning_rates']={g.get('role','all'):g['lr'] for g in self.optimizer.param_groups}
            result.update(engine_id=self.engine_id,cold_load=cold,precision=precision,micro_batch=micro_batch,
                          request_sha256=request_hash,output_sha256=digest(out),
                          timings={'load_seconds':loaded-start,'update_seconds':trained-loaded,
                                   'save_seconds':time.monotonic()-trained})
            if prune_checkpoint:
                path.unlink()
                path.with_suffix('.optimizer.pt').unlink(missing_ok=True)
            metadata.write_text(json.dumps(result,indent=2)+'\n')
            save_end=time.monotonic();commit()
            result['timings'].update(commit_seconds=time.monotonic()-save_end,total_seconds=time.monotonic()-start)
            # The coordinator durably records this complete timing report.
            self.current=out
            return result
        except Exception:
            # Failed updates must never be reused as if they were input weights.
            self.current=None
            self.reference_path=None
            raise
