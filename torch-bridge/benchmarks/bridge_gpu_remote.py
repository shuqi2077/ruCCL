import ast
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def verify_bridge(worker, python_files, package_files, model_main):
    import psutil
    work=Path('/kaggle/working')
    started=time.monotonic()
    manifests=[(p.parent,json.loads(p.read_text())) for p in Path('/kaggle/input').rglob('assets/manifest.json')]
    compiled=[(p,m) for p,m in manifests if 'wgpu' in m.get('bridge_features','')]
    models=[(p,m) for p,m in manifests if m.get('model')=='Qwen/Qwen3.5-0.8B']
    assert len(compiled)==len(models)==1
    compiled,manifest=compiled[0];models,model_manifest=models[0]
    assets=work/'combined-assets';assets.mkdir()
    shutil.copy2(compiled/'libruccl_torch_bridge.so',assets/'libruccl_torch_bridge.so')
    assert hashlib.sha256((assets/'libruccl_torch_bridge.so').read_bytes()).hexdigest()==manifest['bridge_sha256']
    for name in ('model','wikitext-train.parquet'):
        (assets/name).symlink_to(models/name,target_is_directory=name=='model')
    for key in ('model','model_revision','data_revision'):manifest[key]=model_manifest[key]
    (assets/'manifest.json').write_text(json.dumps(manifest))
    source=work/'portable-source'
    for name,content in package_files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    for name,content in python_files.items():
        path=source/'crates/ruda-torch/python/ruda_torch'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    subprocess.run(['python','-m','pip','install','--no-cache-dir','--no-deps','--no-build-isolation',
                    str(source/'crates/ruCCL/torch-bridge/python')],check=True)
    env=os.environ.copy();env['RUCCL_TORCH_LIBRARY']=str(assets/'libruccl_torch_bridge.so')
    tree=ast.parse(worker)
    checks=ast.get_source_segment(worker,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bridge_checks'))
    cuda_program='''import json,os,torch
import torch.distributed as dist
from ruda_collectives import ReplicaGroup
RANK=int(os.environ['RANK']);WORLD=int(os.environ['WORLD_SIZE'])
DEVICE=int(os.environ['LOCAL_RANK'])%torch.cuda.device_count()
torch.cuda.set_device(DEVICE);torch.set_num_threads(1)
def event(*args,**kwargs):pass
'''+checks+'''
dist.init_process_group('gloo')
group=ReplicaGroup(device_type='cuda',transport='ruccl',ruccl_algorithm=os.environ['RUCCL_TEST_ALGORITHM'])
bridge_checks(group)
ranks=group.gather_metadata(dict(rank=RANK,device=DEVICE,checks=7,communication=group._ruccl.stats()))
if RANK==0:print('RUCCL_RESULT '+json.dumps(dict(status='passed',ranks=ranks)),flush=True)
group._ruccl.close();dist.destroy_process_group()
'''
    cuda_path=work/'cuda-bridge-check.py';cuda_path.write_text(cuda_program)
    cases=[]
    for ranks in (2,3):
        for algorithm in ('ring','auto'):
            cases.append((f'cuda_{ranks}logical_{algorithm}',
                ['python','-m','torch.distributed.run','--standalone','--nproc-per-node='+str(ranks),str(cuda_path)],algorithm))
    for algorithm in ('ring','auto'):
        cases.append(('vulkan_'+algorithm,['python','-m','ruccl_bridge.benchmark','--backend','vulkan',
            '--devices','discrete:0,discrete:1','--algorithm',algorithm],algorithm))
    reports=[];durations=[];previous_threads={};checkpoint_time=time.time();stage_started=time.monotonic()
    state=dict(stage='native_collectives',completed=0,total=len(cases),unit='actual collective cases',
        recovery='completed case reports retained; interrupted bounded case reruns from compiled library and source fixtures',
        checkpoint=str(assets),remaining_seconds_range=[30,600],estimate_basis='six small primitive cases; previous two-GPU CUDA primitive checks; first Vulkan shader compilation unknown')
    def publish(process=None):
        processes=[];threads=[]
        if process is not None and process.poll() is None:
            try:processes=[psutil.Process(process.pid)]+psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:pass
        try:
            for process in processes:
                for thread in process.threads():
                    key=(process.pid,thread.id);value=thread.user_time+thread.system_time
                    if value>previous_threads.get(key,value):threads.append(key)
                    previous_threads[key]=value
        except psutil.Error:pass
        state.update(total_elapsed_seconds=time.monotonic()-started,
            stage_elapsed_seconds=time.monotonic()-stage_started,checkpoint_age_seconds=time.time()-checkpoint_time,
            resources=dict(
            allocated_cores=len(os.sched_getaffinity(0)),cpu_percent=psutil.cpu_percent(),
            memory=psutil.virtual_memory()._asdict(),swap=psutil.swap_memory()._asdict(),
            active_process_ids=[p.pid for p in processes],active_threads=len(threads),active_thread_ids=threads,
            cpu_per_core=psutil.cpu_percent(percpu=True),
            gpus=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader'],capture_output=True,text=True).stdout.splitlines(),
            disk_free=shutil.disk_usage(work).free))
        (work/'bridge-state.json').write_text(json.dumps(state))
        (work/'bridge-progress.html').write_text('<meta http-equiv="refresh" content="10"><pre>'+html.escape(json.dumps(state,indent=2))+'</pre>')
        print('QWEN_STATE '+json.dumps(state),flush=True)
    publish()
    for name,command,algorithm in cases:
        stage_started=time.monotonic()
        env['RUCCL_TEST_ALGORITHM']=algorithm;state['stage']=name
        path=work/(name+'.log')
        with path.open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                publish(process)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        text=path.read_text()
        result=[json.loads(line[len('RUCCL_RESULT '):]) for line in text.splitlines() if line.startswith('RUCCL_RESULT ')]
        passed=process.returncode==0 and len(result)==1 and result[0]['status']=='passed'
        report=dict(case=name,status='passed' if passed else 'failed',
                    result=result[0] if result else None,error=None if passed else text[-4000:])
        reports.append(report)
        durations.append(time.monotonic()-stage_started);checkpoint_time=time.time()
        (work/'bridge-report.json').write_text(json.dumps(reports,indent=2))
        remaining=len(cases)-len(reports)
        state.update(completed=len(reports),checkpoint=str(work/'bridge-report.json'),recent_rate=1/durations[-1],
            remaining_seconds_range=[min(durations)*remaining,max(durations)*3*remaining],estimate_basis='actual completed primitive case wall times; widened for first Vulkan shader compilation')
        print('BRIDGE_CASE '+json.dumps(report),flush=True);publish()
    model_main(worker,python_files,assets_path=assets,started_at=started)
    print('BRIDGE_RESULT '+json.dumps(dict(cases=reports,
        status='passed' if all(r['status']=='passed' for r in reports) else 'partial',
        hardware_scope='CUDA and Vulkan on the allocated NVIDIA GPUs; 3 logical CUDA ranks share 2 physical GPUs; no ROCm or Metal runtime claim')),flush=True)
