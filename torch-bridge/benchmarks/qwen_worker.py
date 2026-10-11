import gc
import json
import os
from pathlib import Path
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from ruda_collectives import ReplicaGroup

ASSETS=Path(os.environ['QWEN_ASSETS'])
OUTPUT=Path('/kaggle/working/results')
RANK=int(os.environ['RANK'])
WORLD=int(os.environ['WORLD_SIZE'])
DEVICE=int(os.environ['LOCAL_RANK'])
CASE=os.environ['QWEN_CASE']
SEQUENCE=int(os.environ.get('QWEN_SEQUENCE','64'))
STEPS=4
DECODE=16


def event(stage,completed,total,**extra):
    if RANK==0: print('QWEN_EVENT '+json.dumps(dict(case=CASE,stage=stage,completed=completed,total=total,**extra)),flush=True)


def load():
    torch.manual_seed(431)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(ASSETS/'model',dtype=torch.float32,
        device_map={'':DEVICE},attn_implementation='sdpa',local_files_only=True)
    model.model.visual.requires_grad_(False)
    return model


def inputs(tokenizer):
    import pandas as pd
    text='\n'.join(t for t in pd.read_parquet(ASSETS/'wikitext-train.parquet')['text'].tolist()[:100] if t.strip())
    ids=tokenizer(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
    start=(RANK if CASE.startswith('train') else 0)*SEQUENCE
    batch=torch.tensor([ids[start:start+SEQUENCE]],device=DEVICE)
    assert batch.shape==(1,SEQUENCE)
    return batch


def difference(after,before):
    return {key:after[key]-before[key] for key in after if isinstance(after[key],(int,float))}


def bridge_checks(group):
    x=torch.arange(257,device=DEVICE,dtype=torch.float32)+RANK*100
    group.sum_(x)
    offset=100*WORLD*(WORLD-1)//2
    torch.testing.assert_close(x,torch.arange(257,device=DEVICE,dtype=torch.float32)*WORLD+offset,rtol=0,atol=0)
    x=torch.arange(3*WORLD,device=DEVICE,dtype=torch.float32)+RANK*100
    gathered=group.all_gather(x)
    torch.testing.assert_close(gathered,torch.cat([torch.arange(3*WORLD,device=DEVICE,dtype=torch.float32)+100*rank for rank in range(WORLD)]),rtol=0,atol=0)
    reduced=group.reduce_scatter(x)
    torch.testing.assert_close(reduced,torch.arange(RANK*3,RANK*3+3,device=DEVICE,dtype=torch.float32)*WORLD+offset,rtol=0,atol=0)
    group.broadcast_(x,root=WORLD-1)
    torch.testing.assert_close(x,torch.arange(3*WORLD,device=DEVICE,dtype=torch.float32)+100*(WORLD-1),rtol=0,atol=0)
    for op,extreme in ((group.max_,100*(WORLD-1)),(group.min_,0)):
        y=torch.arange(7,device=DEVICE,dtype=torch.float32)+RANK*100
        op(y);torch.testing.assert_close(y,torch.arange(7,device=DEVICE,dtype=torch.float32)+extreme,rtol=0,atol=0)
    y=torch.arange(13,device=DEVICE,dtype=torch.float32)+RANK*100
    group._ruccl.sum_async(y).wait()
    torch.testing.assert_close(y,torch.arange(13,device=DEVICE,dtype=torch.float32)*WORLD+offset,rtol=0,atol=0)
    event('bridge_verified',7,7)


class ParallelMLP(torch.nn.Module):
    def __init__(self,mlp,group):
        super().__init__();self.group=group;self.act_fn=mlp.act_fn
        total=mlp.gate_proj.out_features
        begin=RANK*total//WORLD
        width=(RANK+1)*total//WORLD-begin
        self.gate=torch.nn.Parameter(mlp.gate_proj.weight[begin:begin+width].detach().clone())
        self.up=torch.nn.Parameter(mlp.up_proj.weight[begin:begin+width].detach().clone())
        self.down=torch.nn.Parameter(mlp.down_proj.weight[:,begin:begin+width].detach().contiguous().clone())
        assert all(layer.bias is None for layer in (mlp.gate_proj,mlp.up_proj,mlp.down_proj))
    def forward(self,value):
        hidden=self.act_fn(torch.nn.functional.linear(value,self.gate))*torch.nn.functional.linear(value,self.up)
        return self.group.sum_(torch.nn.functional.linear(hidden,self.down).contiguous())


def inference(model,batch,group):
    model.eval()
    if group is not None:
        for layer in model.model.language_model.layers:
            layer.mlp=ParallelMLP(layer.mlp,group)
        assert len(model.model.language_model.layers)==24
    rows=[];reference=None
    with torch.inference_mode():
        for step in range(STEPS):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            before={} if group is None else group._ruccl.stats()
            result=model(input_ids=batch,use_cache=True,logits_to_keep=1)
            torch.cuda.synchronize();prefill=time.perf_counter()-start
            last=result.logits[:,-1].float()
            if step==1: reference=last.cpu()
            tokens=[];decode_start=time.perf_counter()
            for token_step in range(DECODE):
                token=result.logits[:,-1].argmax(-1,keepdim=True)
                tokens.append(int(token.item()))
                result=model(input_ids=token,past_key_values=result.past_key_values,use_cache=True,logits_to_keep=1)
                event('decode',step*DECODE+token_step+1,STEPS*DECODE)
            torch.cuda.synchronize();decode=time.perf_counter()-decode_start
            after={} if group is None else group._ruccl.stats()
            rows.append(dict(step=step,warmup=step==0,prefill_seconds=prefill,decode_seconds=decode,
                decode_tokens=DECODE,tokens=tokens,peak_allocated=torch.cuda.max_memory_allocated(),
                communication=difference(after,before),prefill_tokens=SEQUENCE))
            event('inference_iteration',step+1,STEPS,elapsed_seconds=prefill+decode)
    if RANK==0:
        torch.save(reference,OUTPUT/(CASE+'-logits.pt'))
        if group is not None:
            golden=torch.load(OUTPUT/'infer_single-logits.pt',weights_only=True)
            torch.testing.assert_close(reference,golden,rtol=5e-4,atol=5e-4)
            original=json.loads((OUTPUT/'infer_single.json').read_text())
            assert rows[1]['tokens']==original['ranks'][0]['iterations'][1]['tokens']
    return rows


def train(model,batch,group):
    model.train();model.config.use_cache=False
    if group is not None: group.initialize(model,broadcast_buffers=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,fused=True)
    rows=[];count=SEQUENCE-1
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        result=model(input_ids=batch,labels=batch,use_cache=False)
        (result.loss*count).backward()
        torch.cuda.synchronize();compute=time.perf_counter()-start
        before={} if group is None else group._ruccl.stats()
        expected_probes={}
        if group is not None and step==0:
            local_probes={name:p.grad.view(-1)[:min(1024,p.numel())].cpu().tolist()
                          for name,p in model.named_parameters() if p.grad is not None}
            peers=group.gather_metadata(local_probes)
            expected_probes={name:sum(torch.tensor(peer[name]) for peer in peers)/(count*WORLD)
                             for name in local_probes}
        communication_start=time.perf_counter()
        torch.cuda.empty_cache()
        if group is None:
            for p in model.parameters():
                if p.grad is not None:p.grad.div_(count)
        else:
            group.synchronize_gradients(local_weight=count,bucket_bytes=16*1024*1024 if CASE=='train_bucketed' else None)
        torch.cuda.synchronize();communication=time.perf_counter()-communication_start
        after={} if group is None else group._ruccl.stats()
        # Independent CPU check on actual gradients, before the optimizer update.
        if group is not None and step==0:
            for name,p in model.named_parameters():
                if p.grad is not None:
                    probe=p.grad.view(-1)[:min(1024,p.numel())].cpu()
                    torch.testing.assert_close(probe,expected_probes[name],rtol=2e-6,atol=2e-8)
        optimizer_start=time.perf_counter();optimizer.step();torch.cuda.synchronize()
        optimizer_seconds=time.perf_counter()-optimizer_start
        elapsed=time.perf_counter()-start
        loss=float(result.loss.detach())
        assert torch.isfinite(result.loss).item()
        row=dict(step=step,warmup=step==0,loss=loss,compute_seconds=compute,
            gradient_sync_seconds=communication,optimizer_seconds=optimizer_seconds,step_seconds=elapsed,
            tokens=count*WORLD,communication=difference(after,before),peak_allocated=torch.cuda.max_memory_allocated())
        rows.append(row);event('training_step',step+1,STEPS,elapsed_seconds=elapsed)
        del result
    # Identical initialization and inputs, equivalent reduction/optimizer semantics.
    if CASE=='train_bucketed' and RANK==0:
        golden=json.loads((OUTPUT/'train_unbucketed.json').read_text())['ranks'][0]['iterations']
        for actual,expected in zip(rows,golden):
            assert abs(actual['loss']-expected['loss'])<=max(2e-5,abs(expected['loss'])*2e-5)
    return rows


def main():
    OUTPUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.cuda.set_device(DEVICE)
    torch.backends.cuda.matmul.allow_tf32=False
    assert DEVICE < torch.cuda.device_count()
    group=None
    if WORLD>1:
        dist.init_process_group('gloo',timeout=timedelta(seconds=600))
        group=ReplicaGroup(device_type='cuda',transport='ruccl',ruccl_algorithm='auto' if CASE.endswith('optimized') else 'ring')
        bridge_checks(group)
    tokenizer=AutoTokenizer.from_pretrained(ASSETS/'model',local_files_only=True)
    batch=inputs(tokenizer);model=load()
    event('model_loaded',1,1,parameters=sum(p.numel() for p in model.parameters()),
          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),dtype='float32',device=DEVICE)
    iterations=train(model,batch,group) if CASE.startswith('train') else inference(model,batch,group)
    payload=dict(rank=RANK,device=DEVICE,iterations=iterations,communication=None if group is None else group._ruccl.stats())
    ranks=[payload] if group is None else group.gather_metadata(payload)
    if RANK==0:
        report=dict(case=CASE,world_size=WORLD,sequence=SEQUENCE,ranks=ranks,status='passed',
            model='Qwen/Qwen3.5-0.8B',dtype='float32',scope='text; full language model; FFN-only tensor parallel inference',
            checkpoint='case boundary; restart current bounded case from pinned pretrained weights and fresh AdamW; not mid-step optimizer resume')
        temporary=OUTPUT/(CASE+'.next.json');temporary.write_text(json.dumps(report));temporary.replace(OUTPUT/(CASE+'.json'))
        print('QWEN_CASE '+json.dumps(report),flush=True)
    if group is not None:group._ruccl.close();dist.destroy_process_group()
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
