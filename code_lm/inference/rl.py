"""Persistent vLLM rollout service with in-place checkpoint synchronization."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

def reload_model(model, checkpoint, expected_hash):
    """Called inside the vLLM worker, after every previous request has finished."""
    import torch
    from code_lm.inference.export import expand_weights, materialize_loop
    with open(checkpoint, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected_hash:
        raise ValueError('Checkpoint changed during synchronization')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    from code_lm.model import model_args_from_config, inference_weights
    metadata=saved.get('config') or {}
    source=saved.get('model_args') or metadata.get('model_args') or metadata
    state,args=inference_weights(saved['model'],model_args_from_config(source))
    if args!=model_args_from_config(model.config.code_lm_source_model_args):
        raise ValueError('Cannot hot-swap a different architecture')
    order = model.config.code_lm_source_layer_order
    weights = (materialize_loop(state, args)[0].items() if args.get('loop_lora_rank')
               else expand_weights(state, order))
    loaded = model.load_weights(weights)
    torch.cuda.synchronize()
    return {'checkpoint_sha256': digest, 'loaded_tensors': len(loaded)}


class WeightSyncWorker:
    def reload_bst_checkpoint(self, checkpoint, expected_hash):
        return reload_model(self.model_runner.model, checkpoint, expected_hash)


class Sampler:
    def __init__(self, tokenizer_dir=Path('data/tokenizer'), max_tokens=256, eager=False):
        tokenizer_dir = Path(tokenizer_dir)
        self.max_tokens, self.eager = max_tokens, eager
        from code_lm.tokenizer import Tokenizer
        self.llm = None
        self.digest = None
        self.engine_id = str(uuid.uuid4())
        self.sync_count = 0
        self.tok = Tokenizer.from_files(tokenizer_dir/'vocab.json',
                                       tokenizer_dir/'merges.txt', ['<|endoftext|>'])
        self.eos = self.tok.token_to_id[b'<|endoftext|>']

    def sync(self, checkpoint):
        from vllm import LLM
        from code_lm.inference import register
        from code_lm.inference.export import export_checkpoint
        from code_lm.inference.generate import engine_options
        import torch
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise ValueError('Invalid checkpoint path')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        start = time.monotonic()
        if self.llm is None:
            register()
            self.directory = tempfile.TemporaryDirectory(prefix='bst-rl-serving-')
            exported = Path(self.directory.name) / 'model'
            manifest = export_checkpoint(path, exported, self.eos)
            self.args = manifest['model_args']
            options = engine_options(exported, self.args['context_length'], eager=self.eager)
            options.update(max_num_seqs=64, kv_cache_memory_bytes=2*1024**3,
                           worker_extension_cls='code_lm.inference.rl.WeightSyncWorker')
            options['compilation_config']['cudagraph_capture_sizes'] = [] if self.eager else [1,2,4,8,16,32,64]
            self.llm = LLM(**options)
            self.sync_count += 1
        elif self.digest != digest:
            # No generation is in flight: synchronous method, concurrency=1.
            result = self.llm.collective_rpc('reload_bst_checkpoint', args=(str(path), digest))
            if len(result)!=1 or result[0]['checkpoint_sha256']!=digest:
                raise RuntimeError('Weight sync did not acknowledge the expected policy')
            self.llm.reset_prefix_cache()
            self.sync_count += 1
        self.digest = digest
        return time.monotonic()-start, torch.cuda.get_device_name(0)

    def generate(self, rows, checkpoint, output, samples=1, temperature=0., seed=42):
        from vllm import SamplingParams
        from code_lm.inference.generate import sampling_options
        if not 1<=len(rows)<=256 or not 1<=samples<=8:
            raise ValueError('Invalid batch size')
        sync_seconds, gpu = self.sync(checkpoint)
        out = Path(output).resolve()
        if out.exists():
            raise ValueError('Output path already exists')
        prompt_ids = [self.tok.encode(r['prompt']) for r in rows]
        if any(len(p)+self.max_tokens>self.args['context_length'] for p in prompt_ids):
            raise ValueError('Context overflow')
        options = sampling_options(self.eos,self.max_tokens,temperature,1.,samples,seed)
        options['logprobs'] = 0  # Includes the sampled token, even with zero alternatives.
        start = time.monotonic()
        outputs = self.llm.generate([{'prompt_token_ids':p} for p in prompt_ids], SamplingParams(**options), use_tqdm=False)
        elapsed = time.monotonic()-start
        results=[]
        for row,p,request in zip(rows,prompt_ids,outputs,strict=True):
            for j,answer in enumerate(request.outputs):
                tokens=list(answer.token_ids)
                if not tokens or len(tokens)!=len(answer.logprobs):
                    raise RuntimeError('Missing token-aligned behavior log probabilities')
                behavior=[float(scores[token].logprob) for token,scores in zip(tokens,answer.logprobs,strict=True)]
                eos_finished=answer.finish_reason=='stop' and answer.stop_reason==self.eos
                if eos_finished and tokens[-1]!=self.eos:
                    raise RuntimeError('Engine omitted EOS from token/log-probability stream')
                text_tokens=tokens[:-1] if tokens[-1]==self.eos else tokens
                results.append({'id':row['id']+f'-sample{j}' if samples>1 else row['id'],
                                'source_id':row['id'],'sample_index':j,'completion':self.tok.decode(text_tokens),
                                'finish_reason':'eos' if eos_finished else answer.finish_reason,
                                'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
                                'checkpoint_sha256':self.digest,'prompt_token_ids':p,'answer_token_ids':tokens,
                                'behavior_log_probs':behavior,'temperature':temperature,'seed':seed,
                                'backend':'vllm','engine_id':self.engine_id,'sync_count':self.sync_count})
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(''.join(json.dumps(r)+'\n' for r in results))
        stats={'engine_id':self.engine_id,'sync_count':self.sync_count,'gpu':gpu,
               'checkpoint_sha256':self.digest,'sync_seconds':sync_seconds,'generation_seconds':elapsed,
               'answers':len(results),'answer_tokens':sum(len(r['answer_token_ids']) for r in results)}
        out.with_suffix('.metrics.json').write_text(json.dumps(stats,indent=2))
        print(json.dumps(stats),flush=True)
        return results


class TorchSampler:
    """Portable CPU/CUDA/MPS sampler; records exact on-policy token log probabilities."""
    def __init__(self, tokenizer_dir=Path('data/tokenizer'), device='cpu', max_tokens=256, batch_size=8):
        from code_lm.data.prepare import load_tokenizer
        self.tok = load_tokenizer(Path(tokenizer_dir))
        self.eos = self.tok.token_to_id[b'<|endoftext|>']
        self.device, self.max_tokens, self.batch_size = device, max_tokens, batch_size
        self.model, self.digest = None, None

    def generate(self, rows, checkpoint, output, samples=1, temperature=0., seed=42):
        import torch
        from code_lm.training.sft import load_model
        from code_lm.training.grpo import log_probs
        from code_lm.inference.torch_generate import generate_batch
        out = Path(output)
        if out.exists(): raise FileExistsError(out)
        with Path(checkpoint).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != self.digest:
            self.model = None
            self.model, self.args, _ = load_model(Path(checkpoint), torch.device(self.device))
            self.model.eval()
            self.digest = digest
        if not rows or not 1 <= samples <= 8 or temperature not in (0., 1.):
            raise ValueError('Require 1–8 samples and temperature 0 or 1')
        expanded = [(r, j) for r in rows for j in range(samples)]
        results = []
        for start in range(0, len(expanded), self.batch_size):
            batch = expanded[start:start+self.batch_size]
            prompts = [self.tok.encode(row['prompt']) for row, _ in batch]
            answers = generate_batch(self.model, prompts, max_new_tokens=self.max_tokens,
                eos_id=self.eos, context_length=self.args['context_length'], temperature=temperature, seed=seed+start)
            records = []
            for (row, j), prompt, answer in zip(batch, prompts, answers, strict=True):
                tokens = list(answer['token_ids'])
                if answer['finish_reason'] == 'eos': tokens.append(self.eos)
                records.append({'id': row['id']+f'-sample{j}' if samples>1 else row['id'],
                    'source_id':row['id'], 'sample_index':j, 'completion':self.tok.decode(answer['token_ids']),
                    'finish_reason':answer['finish_reason'], 'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
                    'checkpoint_sha256':digest, 'prompt_token_ids':prompt, 'answer_token_ids':tokens,
                    'temperature':temperature, 'seed':seed, 'backend':'torch'})
            with torch.inference_mode():
                for record, lp in zip(records, log_probs(self.model, records), strict=True):
                    record['behavior_log_probs'] = lp.cpu().tolist()
            results.extend(records)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as stream:
            stream.write(''.join(json.dumps(row)+'\n' for row in results))
        return results
