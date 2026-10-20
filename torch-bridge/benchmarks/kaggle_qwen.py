"""Remote-only build/model cache/results; local temporary upload files are removed."""
import ast
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
API=KaggleApi();API.authenticate()
USER=API.get_config_value('username')
RUN=uuid.uuid4().hex[:6]
FILES={name:(ROOT/name).read_text(encoding='utf-8') for name in (
    'crates/ruCCL/torch-bridge/Cargo.toml','crates/ruCCL/torch-bridge/src/lib.rs',
    'crates/ruCCL/torch-bridge/src/cuda.rs','crates/ruCCL/torch-bridge/src/tensor.rs',
    'crates/ruCCL/src/tensor_device/device.rs',
    'crates/ruCCL/torch-bridge/python/ruccl_bridge/__init__.py',
    'crates/ruCCL/torch-bridge/python/ruccl_bridge/benchmark.py',
    'crates/ruCCL/torch-bridge/python/pyproject.toml','crates/ruCCL/torch-bridge/python/setup.py',
    'crates/ruCCL/torch-bridge/python/ruda_collectives/__init__.py',
    'crates/ruda-torch/python/ruda_torch/distributed_training.py',
    'crates/ruda-torch/python/ruda_torch/ruccl_transport.py',
    'crates/ruda-torch/python/ruda_torch/parallel_training.py',
    'crates/ruda-torch/python/ruda_torch/parallel_adapters.py')}


def submit(kind,program,source=None,accelerator=None):
    title='Ruccl Qwen08 '+kind+' 20261011 '+RUN
    kernel=USER+'/'+title.lower().replace(' ','-')
    with tempfile.TemporaryDirectory(prefix='ruccl-upload-') as directory:
        path=Path(directory);(path/'run.py').write_text(program,encoding='utf-8')
        metadata=dict(id=kernel,title=title,code_file='run.py',language='python',kernel_type='script',is_private=True,
            enable_gpu=kind=='gpu',enable_internet=True,dataset_sources=[],competition_sources=[],
            kernel_sources=[] if source is None else [source],model_sources=[])
        selected=(accelerator or os.environ.get('KAGGLE_GPU_ACCELERATOR','NvidiaTeslaT4')) if kind=='gpu' else None
        if kind=='gpu':metadata['machine_shape']=selected
        (path/'kernel-metadata.json').write_text(json.dumps(metadata))
        API.kernels_push(str(path),timeout='1800',acc=selected)
    print('KERNEL '+kernel,flush=True)
    return kernel


def monitor(kernel):
    print('TERMINAL PANEL: stage | real completed/total | resources | elapsed | checkpoint | remaining range',flush=True)
    seen=0
    while True:
        status=str(API.kernels_status(kernel).status).split('.')[-1].lower()
        try:
            if status=='queued':
                print('queued: resource allocation unknown; no work completed',flush=True);time.sleep(10);continue
            index=0
            for event in API.kernels_logs_stream(kernel):
                index+=1
                if index<=seen:continue
                seen=index
                data=event.get('data','')
                for line in data.splitlines():
                    if line.startswith(('QWEN_STATE ','QWEN_EVENT ','QWEN_RESULT ','QWEN_FAILURE ','QWEN_CASE ')):
                        print(line,flush=True)
                    elif any(word in line.lower() for word in ('error','failed','traceback','panic')):
                        print(line,flush=True)
        except Exception as error:
            print('stream diagnostic: '+type(error).__name__+': '+str(error)[:160],flush=True)
        status=str(API.kernels_status(kernel).status).split('.')[-1].lower()
        if any(word in status for word in ('complete','failed','error','cancel')):
            logs=API.kernels_logs(kernel)
            try:logs=''.join(event.get('data','') for event in json.loads(logs))
            except ValueError:pass
            if 'QWEN_FAILURE ' in logs or 'complete' not in status:
                for line in logs.splitlines():
                    if line.startswith(('QWEN_FAILURE ','error[')):print(line[:1000],flush=True)
                raise RuntimeError('remote job failed: '+kernel)
            return logs
        time.sleep(10)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        build=sys.argv[1]
        monitor(build)
    else:
        program=(HERE/'prepare_remote.py').read_text(encoding='utf-8')+'\ntry:\n    prepare('+repr(FILES)+')\nexcept Exception as error:\n    print("QWEN_FAILURE "+str(error),flush=True)\n    raise\n'
        ast.parse(program)
        build=submit('prepare',program);monitor(build)
    python_files={name:(ROOT/'crates/ruda-torch/python/ruda_torch'/name).read_text(encoding='utf-8')
                  for name in ('distributed_training.py','ruccl_transport.py','parallel_training.py','parallel_adapters.py')}
    previous=None
    if len(sys.argv)>2:
        old_kernel=sys.argv[2]
        logs=API.kernels_logs(old_kernel)
        try:logs=''.join(event.get('data','') for event in json.loads(logs))
        except ValueError:pass
        reports={}
        for line in logs.splitlines():
            if line.startswith('QWEN_CASE '):
                report=json.loads(line[len('QWEN_CASE '):])
                if report['status']=='passed' and report['case'].startswith('infer_'):
                    reports[report['case']]=report
        if len(reports)!=3:raise RuntimeError('missing three valid inference checkpoints; refusing to invent completed cases')
        previous=dict(kernel=old_kernel,reports=reports)
        print('Recovering three actual passed inference cases; rerunning only training.',flush=True)
    program=(HERE/'gpu_remote.py').read_text(encoding='utf-8')+'\nmain('+repr((HERE/'qwen_worker.py').read_text(encoding='utf-8'))+', '+repr(python_files)+', '+repr(previous)+')\n'
    ast.parse(program)
    kernel=submit('gpu',program,build)
    logs=monitor(kernel)
    results=[line[len('QWEN_RESULT '):] for line in logs.splitlines() if line.startswith('QWEN_RESULT ')]
    if not results:raise RuntimeError('missing actual model report')
    print('FINAL '+results[-1],flush=True)
    print('Remote report: https://www.kaggle.com/code/'+kernel+'/output',flush=True)
