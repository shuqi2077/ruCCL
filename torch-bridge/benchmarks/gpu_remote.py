import hashlib
import html
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import traceback

TRANSFORMERS='536ecc007387a50e77603bb5d92100e9b07514cc'


def main(worker,python_files,previous=None,assets_path=None,started_at=None):
    import psutil
    work=Path('/kaggle/working');output=work/'results';output.mkdir(exist_ok=True)
    started=time.monotonic() if started_at is None else started_at;stage_started=time.monotonic()
    state=dict(stage='setup',completed=0,total=6,unit='actual Qwen benchmark cases',
        recent_rate='unknown',remaining_seconds_range=[60,1800],estimate_basis='6 bounded cases; initial performance preflight follows',
        checkpoint=None,checkpoint_age_seconds=None,recovery='completed cases retained; replay interrupted case from pinned pretrained weights and fresh optimizer; no mid-step optimizer resume')
    active=None;previous_threads={};last_checkpoint=None
    def publish():
        nonlocal previous_threads
        threads=[];processes=[]
        if active is not None and active.poll() is None:
            try:
                processes=[psutil.Process(active.pid)]+psutil.Process(active.pid).children(recursive=True)
                for process in processes:
                    for thread in process.threads():
                        key=(process.pid,thread.id);value=thread.user_time+thread.system_time
                        if value>previous_threads.get(key,value):threads.append(key)
                        previous_threads[key]=value
            except psutil.Error:pass
        now=time.monotonic()
        resources=dict(allocated_cores=len(os.sched_getaffinity(0)),cpu_percent=psutil.cpu_percent(),
            cpu_per_core=psutil.cpu_percent(percpu=True),active_threads=len(threads),active_thread_ids=threads,
            active_processes=len(processes),active_process_ids=[process.pid for process in processes],
            memory=psutil.virtual_memory()._asdict(),swap=psutil.swap_memory()._asdict(),disk_free=shutil.disk_usage(work).free)
        query=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
        resources['gpus']=query.stdout.strip().splitlines()
        state.update(total_elapsed_seconds=now-started,stage_elapsed_seconds=now-stage_started,resources=resources,
            checkpoint_age_seconds=None if last_checkpoint is None else time.time()-last_checkpoint)
        text=json.dumps(state)
        temporary=work/'state.next.json';temporary.write_text(text);temporary.replace(work/'state.json')
        (work/'progress.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10"><h1>ruCCL Qwen 0.8B</h1><pre>'+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>')
        print('QWEN_STATE '+text,flush=True)
    def run(command,env=None):
        nonlocal active
        active=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        lines=queue.Queue()
        def read():
            for line in active.stdout:lines.put(line)
            lines.put(None)
        threading.Thread(target=read,daemon=True).start()
        ended=False;sample=0
        with (work/'execution.log').open('a') as log:
            while not ended or active.poll() is None:
                try:
                    line=lines.get(timeout=1)
                    if line is None:ended=True
                    else:
                        log.write(line);log.flush()
                        if line.startswith('QWEN_EVENT '):state['case_progress']=json.loads(line[len('QWEN_EVENT '):])
                        if len(line)<8000:print(line.rstrip(),flush=True)
                except queue.Empty:pass
                if time.monotonic()>=sample:publish();sample=time.monotonic()+10
        if active.wait()!=0:raise RuntimeError('remote command failed: '+str(command[:4]))
    try:
        publish()
        import torch
        world=int(os.environ.get('QWEN_WORLD_SIZE',torch.cuda.device_count()))
        if not 1 <= world <= torch.cuda.device_count():
            raise RuntimeError('QWEN_WORLD_SIZE must select available physical CUDA GPUs')
        assets_list=([Path(assets_path)/'manifest.json'] if assets_path is not None
            else list(Path('/kaggle/input').rglob('assets/manifest.json')))
        if len(assets_list)!=1:raise RuntimeError('missing or ambiguous remote prepared assets')
        assets=assets_list[0].parent;manifest=json.loads(assets_list[0].read_text())
        library=assets/'libruccl_torch_bridge.so'
        assert hashlib.sha256(library.read_bytes()).hexdigest()==manifest['bridge_sha256']
        package=work/'ruda_collectives';package.mkdir(exist_ok=True)
        (package/'__init__.py').write_text('from .distributed_training import ReplicaGroup\n')
        for name,source in python_files.items():
            (package/name).write_text(source)
        manifest['executed_python_sha256']={name:hashlib.sha256(source.encode()).hexdigest() for name,source in python_files.items()}
        state['checkpoint']=str(assets);last_checkpoint=time.time()
        (work/'manifest.json').write_text(json.dumps(manifest))
        run(['python','-m','pip','install','--no-cache-dir','--quiet',
             'https://github.com/huggingface/transformers/archive/'+TRANSFORMERS+'.zip','accelerate'])
        worker_path=work/'qwen_worker.py';worker_path.write_text(worker)
        env=os.environ.copy();env.update(QWEN_ASSETS=str(assets),RUCCL_TORCH_LIBRARY=str(library),
            PYTHONPATH=str(work)+':'+str(assets)+':'+env.get('PYTHONPATH',''),TOKENIZERS_PARALLELISM='false',
            HF_HOME='/kaggle/temp/hf-runtime',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTORCH_ALLOC_CONF='expandable_segments:True')
        cases=['infer_single','infer_ring','infer_optimized','train_single','train_unbucketed','train_bucketed']
        if world == 1: cases=['infer_single','train_single']
        state['total']=len(cases)
        resumed={} if previous is None else previous['reports']
        for case,report in resumed.items():
            assert case in cases and case.startswith('infer_') and report['status']=='passed' and report['sequence']==64
            assert report['world_size']==(1 if case.endswith('single') else world)
            assert report['model']==manifest['model'] and report['dtype']=='float32'
            assert all(len(rank['iterations'])==4 for rank in report['ranks'])
            (output/(case+'.json')).write_text(json.dumps(report))
        durations=[]
        for index,case in enumerate(cases):
            if case in resumed:
                state.update(completed=index+1,checkpoint=str(output/(case+'.json')),stage='recovered_'+case)
                last_checkpoint=time.time();publish();continue
            stage_started=time.monotonic();state.update(stage=case,case_progress=None)
            env['QWEN_CASE']=case
            distributed=not case.endswith('single')
            command=['python','-m','torch.distributed.run','--standalone','--nproc-per-node='+str(world if distributed else 1),str(worker_path)]
            publish();run(command,env)
            report=json.loads((output/(case+'.json')).read_text())
            assert report['status']=='passed' and all(len(rank['iterations'])==4 for rank in report['ranks'])
            durations.append(time.monotonic()-stage_started)
            state.update(completed=index+1,checkpoint=str(output/(case+'.json')),recent_rate=1/durations[-1],
                remaining_seconds_range=[min(durations)*(len(cases)-index-1),max(durations)*3*(len(cases)-index-1)],
                estimate_basis='actual completed case wall times; training/inference differ, upper estimate widened')
            last_checkpoint=time.time();publish()
        reports={case:json.loads((output/(case+'.json')).read_text()) for case in cases}
        def metrics(case):
            iterations=reports[case]['ranks'][0]['iterations'][1:]
            if case.startswith('train'):
                seconds=sum(row['step_seconds'] for row in iterations)
                native_seconds=sum(row['communication'].get('wall_seconds',0) for row in iterations)
                return dict(tokens_per_second=sum(row['tokens'] for row in iterations)/seconds,
                    mean_step_seconds=seconds/len(iterations),communication_fraction=native_seconds/seconds,
                    mean_communication_seconds=native_seconds/len(iterations),
                    gradient_sync_stage_fraction=sum(row['gradient_sync_seconds'] for row in iterations)/seconds,
                    mean_gradient_sync_stage_seconds=sum(row['gradient_sync_seconds'] for row in iterations)/len(iterations),
                    max_peak_allocated=max(row['peak_allocated'] for row in iterations))
            return dict(prefill_tokens_per_second=sum(row['prefill_tokens'] for row in iterations)/sum(row['prefill_seconds'] for row in iterations),
                decode_tokens_per_second=sum(row['decode_tokens'] for row in iterations)/sum(row['decode_seconds'] for row in iterations),
                mean_communication_seconds=sum(row['communication'].get('wall_seconds',0) for row in iterations)/len(iterations),
                communication_fraction=sum(row['communication'].get('wall_seconds',0) for row in iterations)/sum(row['prefill_seconds']+row['decode_seconds'] for row in iterations),
                max_peak_allocated=max(row['peak_allocated'] for row in iterations))
        result=dict(status='passed',cases=len(cases),model=manifest['model'],revision=manifest['model_revision'],
            metrics={case:metrics(case) for case in cases},source_revision=manifest['source_revision'],
            scope='real pretrained Qwen3.5-0.8B text training and inference; FP32; sequence64; 1 warmup+3 measured iterations; decode16; no NCCL tensor payload',
            training_optimization='16MiB FP32 gradient buckets',inference_optimization='ruCCL automatic small-message dispatch versus forced Ring; FFN tensor parallelism only',
            memory_scope='max_peak_allocated is PyTorch allocator only; device total memory is sampled in progress, not an exact external CUDA allocation peak',
            measurement_definitions=dict(communication='native ruCCL collective wall time; zero for single GPU; inference covers prefill plus decode',
                gradient_sync_stage='cache release, local normalization, metadata and tensor synchronization; training only'),
            control_transport='Gloo rendezvous/control and untimed correctness samples; actual measured tensor collectives use ruCCL only',
            checkpoints='completed case reports and pinned weights; interrupted case restarts, not mid-step optimizer resume')
        result['resumed_cases_from']=None if previous is None else previous['kernel']
        result['training_memory_policy']='expandable CUDA segments; release inactive cache before forward and synchronization; full FP32 model and optimizer unchanged'
        (output/'report.json').write_text(json.dumps(result,indent=2))
        state.update(stage='complete',completed=len(cases),remaining_seconds_range=[0,0]);publish()
        print('QWEN_RESULT '+json.dumps(result),flush=True)
    except Exception as error:
        (work/'failure.json').write_text(json.dumps(dict(error=str(error),state=state)))
        print('QWEN_FAILURE '+str(error),flush=True);traceback.print_exc();raise
