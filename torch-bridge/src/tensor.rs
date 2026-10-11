use crate::{Error, checked};
use ruccl::{rank::{AlgorithmPolicy, CollectiveTuning, ElementType, ReductionOperation,
    TcpRankSession, UniqueId, communicator::RankCommunicator, device::RankDevice},
    tensor_device::{TensorDevice, TensorBuffer}};
use ruda_tensor::Backend;
use std::{collections::HashMap, ffi::c_void, sync::Mutex, time::{Duration, Instant}};

trait Session: Send {
    fn upload(&mut self, values: &[f32]) -> Result<u64, Error>;
    fn download(&self, id: u64) -> Result<Vec<f32>, Error>;
    fn release(&mut self, id: u64) -> Result<(), Error>;
    fn collective(&mut self, id: u64, operation: u32, root: u32) -> Result<u64, Error>;
    fn stats(&self) -> [u64; 4];
}
struct TensorSession<B: Backend> {
    communicator: RankCommunicator<TensorDevice<B>>,
    buffers: HashMap<u64, TensorBuffer<B, f32>>,
    next_id: u64,
    counters: [u64; 4],
}
impl<B: Backend> TensorSession<B> {
    fn connect(device: B::Device, session: TcpRankSession, ring: bool) -> Result<Self, Error> {
        let mut tuning = CollectiveTuning::from_environment(session.world_size())?;
        if ring { tuning.algorithm_policy = AlgorithmPolicy::Ring; }
        let communicator = RankCommunicator::from_session_with_tuning(
            || Ok::<_, Error>(TensorDevice::<B>::new(device)), session, tuning, "ruccl-tensor-bridge")?;
        Ok(Self { communicator, buffers: HashMap::new(), next_id: 1, counters: [0; 4] })
    }
    fn insert(&mut self, buffer: TensorBuffer<B, f32>) -> Result<u64, Error> {
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or_else(|| Error("buffer ID overflow".into()))?;
        self.buffers.insert(id, buffer);
        Ok(id)
    }
    fn get(&self, id: u64) -> Result<&TensorBuffer<B, f32>, Error> {
        self.buffers.get(&id).ok_or_else(|| Error("unknown or released buffer".into()))
    }
}
impl<B: Backend> Session for TensorSession<B> {
    fn upload(&mut self, values: &[f32]) -> Result<u64, Error> {
        let buffer = self.communicator.execution().upload(values)?;
        self.communicator.execution().synchronize()?;
        self.insert(buffer)
    }
    fn download(&self, id: u64) -> Result<Vec<f32>, Error> {
        Ok(RankDevice::<f32>::copy_from_device(self.communicator.execution(), self.get(id)?)?)
    }
    fn release(&mut self, id: u64) -> Result<(), Error> {
        self.buffers.remove(&id).ok_or_else(|| Error("unknown or released buffer".into()))?;
        Ok(())
    }
    fn collective(&mut self, id: u64, operation: u32, root: u32) -> Result<u64, Error> {
        let buffer = self.get(id)?.clone();
        let collective = self.communicator.device_collective::<f32>(ElementType::F32);
        let started = Instant::now();
        let (output, stats) = match operation {
            0..=2 => {
                let reduction = match operation { 0 => ReductionOperation::Sum,
                    1 => ReductionOperation::Maximum, _ => ReductionOperation::Minimum };
                (None, collective.all_reduce(&buffer, reduction, &reduction)?)
            }
            3 => (None, collective.broadcast(&buffer, root)?),
            4 => { let (output, stats) = collective.all_gather(&buffer)?; (Some(output), stats) },
            5 => { let (output, stats) = collective.reduce_scatter(&buffer,
                ReductionOperation::Sum, &ReductionOperation::Sum)?; (Some(output), stats) },
            _ => return Err(Error("unknown collective".into())),
        };
        self.communicator.execution().synchronize()?;
        self.counters[0] += 1;
        self.counters[1] += stats.transferred_bytes as u64;
        self.counters[2] += stats.reduction_kernel_launches as u64;
        self.counters[3] += started.elapsed().as_nanos() as u64;
        match output { Some(output) => self.insert(output), None => Ok(id) }
    }
    fn stats(&self) -> [u64; 4] { self.counters }
}
struct Group(Mutex<Box<dyn Session>>);

#[cfg(feature = "wgpu")]
fn wgpu_device(backend: wgpu::Backend, class: u32, index: u32)
    -> Result<ruda_tensor_wgpu::WgpuDevice, Error> {
    use ruda_core::future::block_on;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters = block_on(instance.enumerate_adapters(backend.into()));
    adapters.retain(|adapter| match (class, adapter.get_info().device_type) {
        (0 | 4, wgpu::DeviceType::DiscreteGpu) |
        (1 | 4, wgpu::DeviceType::IntegratedGpu) |
        (2 | 4, wgpu::DeviceType::VirtualGpu) => true,
        _ => false,
    });
    if class == 4 {
        if index != 0 { return Err(Error("default GPU index must be zero".into())); }
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        });
    }
    let adapter = adapters.into_iter().nth(index as usize)
        .ok_or_else(|| Error("requested GPU class/index is unavailable".into()))?;
    // Backend slice kernels use integer indexing; no graphics or ray-tracing features are needed.
    let timing = wgpu::Features::TIMESTAMP_QUERY |
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS |
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    let features = adapter.features() & (timing | wgpu::Features::SHADER_INT64);
    eprintln!("ruCCL GPU {:?}; enabled features {:?}", adapter.get_info(), features);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ruCCL FP32 collective device"),
        required_features: features,
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    })).map_err(|e| Error(e.to_string()))?;
    let setup = ruda_tensor_wgpu::WgpuSetup { instance, adapter, device, queue, backend };
    Ok(ruda_tensor_wgpu::init_device(setup, Default::default()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_create(rank: u32, world: u32, backend: u32,
    class: u32, device: u32, port: u32, uid: *const u8, ring: u32, out: *mut *mut c_void) -> i32 {
    checked(|| {
        if uid.is_null() || out.is_null() { return Err(Error("null tensor session argument".into())); }
        let supported = match backend { 2 => cfg!(feature = "rocm"),
            3 => cfg!(all(feature = "wgpu", target_os = "macos")),
            4 => cfg!(feature = "wgpu"), _ => false };
        if !supported { return Err(Error("requested backend is not compiled for this platform".into())); }
        let id = UniqueId::from_bytes(unsafe { std::slice::from_raw_parts(uid, 16) }.try_into().unwrap());
        let session = TcpRankSession::connect(("127.0.0.1", port as u16), id, rank, world, Duration::from_secs(300))?;
        let execution: Box<dyn Session> = match backend {
            #[cfg(feature = "rocm")]
            2 => Box::new(TensorSession::<ruda_tensor_rocm::Rocm<f32>>::connect(
                ruda_tensor_rocm::RocmDevice { index: device as usize }, session, ring != 0)?),
            #[cfg(all(feature = "wgpu", target_os = "macos"))]
            3 => {
                let device = wgpu_device(wgpu::Backend::Metal, class, device)?;
                Box::new(TensorSession::<ruda_tensor_wgpu::Wgpu<f32>>::connect(device, session, ring != 0)?)
            }
            #[cfg(feature = "wgpu")]
            4 => {
                let device = wgpu_device(wgpu::Backend::Vulkan, class, device)?;
                Box::new(TensorSession::<ruda_tensor_wgpu::Wgpu<f32>>::connect(device, session, ring != 0)?)
            }
            _ => return Err(Error("backend unavailable".into())),
        };
        unsafe { *out = Box::into_raw(Box::new(Group(Mutex::new(execution)))).cast(); }
        Ok(())
    })
}
unsafe fn group<'a>(handle: *mut c_void) -> Result<&'a Group, Error> {
    if handle.is_null() { return Err(Error("null tensor session".into())); }
    Ok(unsafe { &*handle.cast::<Group>() })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_upload(handle: *mut c_void, values: *const f32,
    count: usize, out: *mut u64) -> i32 {
    checked(|| {
        if out.is_null() || (count > 0 && values.is_null()) { return Err(Error("null upload argument".into())); }
        let values = if count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(values, count) } };
        let id = unsafe { group(handle) }?.0.lock().map_err(|_| Error("poisoned session".into()))?.upload(values)?;
        unsafe { *out = id; }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_download(handle: *mut c_void, id: u64,
    out: *mut f32, count: usize) -> i32 {
    checked(|| {
        let values = unsafe { group(handle) }?.0.lock().map_err(|_| Error("poisoned session".into()))?.download(id)?;
        if count != values.len() || (count > 0 && out.is_null()) { return Err(Error("download length/pointer mismatch".into())); }
        if count > 0 { unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, count); } }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_collective(handle: *mut c_void, id: u64,
    operation: u32, root: u32, out: *mut u64) -> i32 {
    checked(|| {
        if out.is_null() { return Err(Error("null collective output".into())); }
        let id = unsafe { group(handle) }?.0.lock().map_err(|_| Error("poisoned session".into()))?.collective(id, operation, root)?;
        unsafe { *out = id; }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_release(handle: *mut c_void, id: u64) -> i32 {
    checked(|| unsafe { group(handle) }?.0.lock().map_err(|_| Error("poisoned session".into()))?.release(id))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_stats(handle: *mut c_void, out: *mut u64) -> i32 {
    checked(|| {
        if out.is_null() { return Err(Error("null stats output".into())); }
        let values = unsafe { group(handle) }?.0.lock().map_err(|_| Error("poisoned session".into()))?.stats();
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()); }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_tensor_destroy(handle: *mut c_void) {
    if !handle.is_null() { drop(unsafe { Box::from_raw(handle.cast::<Group>()) }); }
}
