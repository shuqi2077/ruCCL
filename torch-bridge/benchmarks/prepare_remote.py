import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request

SOURCE = 'f97d472046de689efcccedb7b256246e6d4fc63b'
MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
TRANSFORMERS = '536ecc007387a50e77603bb5d92100e9b07514cc'


def prepare(files, *, bridge_features='cuda', include_model=True, check_backends=False, target=None):
    import psutil
    started = time.monotonic()
    assets = Path('/kaggle/working/assets'); assets.mkdir(exist_ok=True)
    temporary = Path('/kaggle/temp/ruccl-qwen'); temporary.mkdir(parents=True, exist_ok=True)
    archive = temporary/'source.tar.gz'
    urllib.request.urlretrieve('https://codeload.github.com/shuqi2077/RUDA/tar.gz/'+SOURCE, archive)
    with tarfile.open(archive) as bundle:
        bundle.extractall(temporary, filter='data')
    root = temporary/('RUDA-'+SOURCE)
    for name, content in files.items():
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    package = assets/'ruda_collectives'; package.mkdir()
    (package/'__init__.py').write_text('from .distributed_training import ReplicaGroup\n')
    for name in ('distributed_training.py','ruccl_transport.py','parallel_training.py','parallel_adapters.py'):
        shutil.copy2(root/'ruda-torch/python/ruda_torch'/name, package/name)
    bridge_package=root/'ruCCL/torch-bridge/python/ruccl_bridge'
    if bridge_package.is_dir(): shutil.copytree(bridge_package,assets/'ruccl_bridge')
    env = os.environ.copy()
    env.update(RUSTUP_HOME=str(temporary/'rustup'), CARGO_HOME=str(temporary/'cargo'),
               CARGO_BUILD_JOBS='2', CARGO_INCREMENTAL='0', CARGO_PROFILE_RELEASE_DEBUG='0')
    env['PATH'] = env['CARGO_HOME']+'/bin:'+env['PATH']
    installer = temporary/'rustup-init'
    urllib.request.urlretrieve('https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init', installer)
    installer.chmod(0o755)
    subprocess.run([str(installer),'-y','--profile','minimal','--default-toolchain','1.97.1','--no-modify-path'],env=env,check=True)
    status = dict(stage='compile', completed=0, total=1, unit='release ruCCL bridge', compiled_units=0,
                  checkpoint='uploaded source and exact patch; restart failed build from source',
                  estimated_remaining_seconds=[60,1800], estimate_basis='first release build, two Cargo jobs')
    stop = threading.Event()
    process = None
    previous_threads = {}
    def panel():
        active_threads=[]
        if process is not None and process.poll() is None:
            try:
                for child in [psutil.Process(process.pid)]+psutil.Process(process.pid).children(recursive=True):
                    for item in child.threads():
                        key=(child.pid,item.id);value=item.user_time+item.system_time
                        if value>previous_threads.get(key,value): active_threads.append(key)
                        previous_threads[key]=value
            except psutil.Error: pass
        state = dict(status, elapsed_seconds=time.monotonic()-started, allocated_cores=len(os.sched_getaffinity(0)),
                     cpu_percent=psutil.cpu_percent(), memory=psutil.virtual_memory()._asdict(),
                     swap=psutil.swap_memory()._asdict(), disk_free=shutil.disk_usage('/kaggle/working').free,
                     active_threads=len(active_threads),active_thread_ids=active_threads,gpus=[], scheduled_cargo_jobs=2)
        (assets/'prepare-state.json').write_text(json.dumps(state))
        print('QWEN_STATE '+json.dumps(state),flush=True)
    def sampling():
        while not stop.is_set():
            panel(); stop.wait(10)
    thread = threading.Thread(target=sampling,daemon=True); thread.start()
    try:
        if target is not None:
            if target!='x86_64-pc-windows-gnu': raise ValueError('unsupported remote cross target')
            env['DEBIAN_FRONTEND']='noninteractive'
            env['CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER']='x86_64-w64-mingw32-gcc'
            prerequisites=[['sudo','apt-get','update','-qq'],
                ['sudo','apt-get','install','-y','-qq','gcc-mingw-w64-x86-64-posix'],
                ['rustup','target','add',target]]
            for index,command in enumerate(prerequisites):
                status.update(stage='cross_toolchain',completed=index,total=len(prerequisites),unit='cross compiler prerequisites')
                process=subprocess.Popen(command,env=env)
                if process.wait()!=0: raise RuntimeError('cross toolchain setup failed')
            status.update(stage='compile',completed=0,total=1,unit='Windows Vulkan bridge')
        command=['cargo','build','--release','--message-format=json','--no-default-features','--features',bridge_features]
        if target is not None: command+=['--target',target]
        process = subprocess.Popen(command,
            cwd=root/'ruCCL/torch-bridge',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            try: record=json.loads(line)
            except ValueError:
                print(line.rstrip(),flush=True); continue
            if record.get('reason')=='compiler-artifact': status['compiled_units']+=1
            if record.get('reason')=='compiler-message': print(record['message'].get('rendered',''),flush=True)
        if process.wait()!=0: raise RuntimeError('ruCCL bridge release build failed')
        filename='libruccl_torch_bridge.so' if target is None else 'ruccl_torch_bridge.dll'
    build=root/'ruCCL/torch-bridge/target'
        if target is not None: build=build/target
        binary=build/'release'/filename
        shutil.copy2(binary,assets/filename)
        if target is not None:
            imports=subprocess.run(['x86_64-w64-mingw32-objdump','-p',str(binary)],capture_output=True,text=True,check=True).stdout
            for dependency in ('libgcc_s_seh-1.dll','libwinpthread-1.dll','libstdc++-6.dll'):
                if dependency in imports:
                    resolved=subprocess.run(['x86_64-w64-mingw32-gcc','-print-file-name='+dependency],capture_output=True,text=True,check=True).stdout.strip()
                    shutil.copy2(resolved,assets/dependency)
    shutil.copy2(root/'ruCCL/torch-bridge/Cargo.lock',assets/'Cargo.lock')
        if check_backends:
            commands=[('rocm_check',['cargo','check','--no-default-features','--features','rocm']),
                      ('metal_target',['rustup','target','add','aarch64-apple-darwin']),
                      ('metal_check',['cargo','check','--no-default-features','--features','wgpu','--target','aarch64-apple-darwin'])]
            completed=0
            for stage,command in commands:
                status.update(stage=stage,completed=completed,total=2,unit='ROCm and Metal compile checks',
                    checkpoint='compiled CUDA/Vulkan library and source; interrupted check restarts from source',
                    estimated_remaining_seconds=[30,1200],estimate_basis='two incremental backend checks; no GPU/model task')
                process=subprocess.Popen(command,cwd=root/'ruCCL/torch-bridge',env=env)
                if process.wait()!=0: raise RuntimeError(stage+' failed')
                if stage.endswith('_check'): completed+=1
        if include_model:
            status.update(stage='model_download',completed=0,total=1,unit='pinned Qwen checkpoint',
                          checkpoint='compiled release bridge',estimated_remaining_seconds=[30,600],
                          estimate_basis='one 0.8B checkpoint and tokenizer; network rate unknown')
            from huggingface_hub import snapshot_download
            snapshot_download(MODEL,revision=REVISION,local_dir=str(assets/'model'),
                allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
            url='https://huggingface.co/datasets/Salesforce/wikitext/resolve/'+DATA_REVISION+'/wikitext-2-raw-v1/train-00000-of-00001.parquet'
            urllib.request.urlretrieve(url,assets/'wikitext-train.parquet')
        manifest=dict(source_revision=SOURCE, model=MODEL if include_model else None,
            model_revision=REVISION if include_model else None, data_revision=DATA_REVISION if include_model else None,
            transformers_revision=TRANSFORMERS, rust='1.97.1', profile='release',
            bridge_features=bridge_features,
            target=target,bridge_file=filename,
            bridge_sha256=hashlib.sha256((assets/filename).read_bytes()).hexdigest(),
            patch_sha256={name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()})
        (assets/'manifest.json').write_text(json.dumps(manifest))
        status.update(stage='prepared',completed=1,total=1,unit='compiled bridge' + (' + pinned model/data' if include_model else ''),
                      checkpoint='private CPU kernel assets; reusable without recompilation',
                      estimated_remaining_seconds=[0,0])
    finally:
        stop.set(); thread.join(); panel()
    return root,env
