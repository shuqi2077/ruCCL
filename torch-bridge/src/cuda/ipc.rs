use super::*;
use ruccl::rank::{ANY_RANK, Opcode};
use std::sync::Mutex;

const CAPACITY: usize = 1024 * 1024;

struct IpcGroup {
    context: Arc<Context>,
    session: Arc<TcpRankSession>,
    rank: usize,
    world: usize,
    staging: Buffer,
    peers: Vec<CUdeviceptr>,
    module: CUmodule,
    reductions: [CUfunction; 4],
    stream: CUstream,
    working: Option<Buffer>,
    counters: [u64; 8],
}

// Each collective is serialized by its communicator mutex and binds its CUDA
// context on the submitting worker. P2P operations have independent streams.
unsafe impl Send for IpcGroup {}

struct IpcHandle {
    group: Mutex<IpcGroup>,
    session: Arc<TcpRankSession>,
    context: Arc<Context>,
    rank: u32,
    world: u32,
    p2p_counters: [AtomicU64; 8],
    p2p_send_buffers: Mutex<std::collections::HashMap<u32, Arc<Mutex<Option<Buffer>>>>>,
}

struct TransferStream(CUstream);
impl TransferStream {
    fn new() -> Result<Self, Error> {
        let mut stream = std::ptr::null_mut();
        unsafe { check(cuStreamCreate(&mut stream, CUstream_flags::CU_STREAM_NON_BLOCKING as u32))?; }
        Ok(Self(stream))
    }
    fn synchronize(&self) -> Result<(), Error> {
        unsafe { check(cuStreamSynchronize(self.0)) }
    }
}
impl Drop for TransferStream {
    fn drop(&mut self) {
        unsafe { let _ = cuStreamSynchronize(self.0); let _ = cuStreamDestroy_v2(self.0); }
    }
}

pub(super) fn reduction_ptx() -> String {
    let mut code = ".version 6.0\n.target sm_50\n.address_size 64\n".to_owned();
    for (index, kind, width, registers) in [
        (0, "f32", 4, ".reg .f32 v,w,z;"),
        (1, "f64", 8, ".reg .f64 v,w,z;"),
        (2, "s32", 4, ".reg .b32 v,w,z;"),
        (3, "s64", 8, ".reg .b64 v,w,z;"),
    ] {
        code += &format!(
            r#"
.visible .entry reduce_{index}(.param .u64 src,.param .u64 dst,.param .u32 n,.param .u32 op) {{
 .reg .pred p,q; .reg .b32 i,t,s,l,k; .reg .b64 a,b,x,y; {registers}
 ld.param.u64 a,[src]; ld.param.u64 b,[dst]; ld.param.u32 l,[n]; ld.param.u32 k,[op];
 mov.u32 i,%ctaid.x; mov.u32 t,%ntid.x; mov.u32 s,%tid.x; mad.lo.u32 i,i,t,s;
 setp.ge.u32 p,i,l; @p bra END;
 mul.wide.u32 x,i,{width}; add.u64 x,a,x; mul.wide.u32 y,i,{width}; add.u64 y,b,y;
 ld.global.{kind} v,[x]; ld.global.{kind} w,[y];
 setp.eq.u32 q,k,0; @q bra SUM; setp.eq.u32 q,k,1; @q bra MAX;
 min.{kind} z,v,w; bra STORE;
 SUM: add.{kind} z,v,w; bra STORE;
 MAX: max.{kind} z,v,w;
 STORE: st.global.{kind} [y],z;
 END: ret;
}}
"#
        );
    }
    code
}

impl IpcGroup {
    fn gather(&self, payload: Vec<u8>) -> Result<Vec<u8>, Error> {
        Ok(self
            .session
            .exchange(
                Opcode::AllGather,
                ElementType::U8,
                ANY_RANK,
                payload.len() as u64,
                payload,
            )?
            .payload)
    }

    fn copy(
        &mut self,
        destination: u64,
        source: u64,
        bytes: usize,
        peer: bool,
    ) -> Result<(), Error> {
        if bytes != 0 {
            self.context.bind()?;
            unsafe {
                check(cuMemcpyDtoDAsync_v2(destination, source, bytes, self.stream))?;
            }
            if peer {
                self.counters[1] += bytes as u64;
            }
        }
        Ok(())
    }

    fn reduce(
        &mut self,
        destination: u64,
        source: u64,
        count: usize,
        dtype: u32,
        op: u32,
    ) -> Result<(), Error> {
        if count == 0 {
            return Ok(());
        }
        let index = match dtype {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            _ => return Err(Error("IPC reduction supports F32/F64/I32/I64".into())),
        };
        let mut n = u32::try_from(count).map_err(|e| Error(e.to_string()))?;
        let (mut source, mut destination, mut op) = (source, destination, op);
        let mut args = [
            (&mut source as *mut u64).cast::<c_void>(),
            (&mut destination as *mut u64).cast(),
            (&mut n as *mut u32).cast(),
            (&mut op as *mut u32).cast(),
        ];
        self.context.bind()?;
        unsafe {
            check(cuLaunchKernel(
                self.reductions[index],
                n.div_ceil(256),
                1,
                1,
                256,
                1,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ))?;
            check(cuStreamSynchronize(self.stream))?;
        }
        self.counters[1] += (count * element_bytes(dtype)?) as u64;
        self.counters[2] += 1;
        Ok(())
    }

    fn barrier(&self) -> Result<(), Error> {
        self.context.bind()?;
        unsafe {
            check(cuStreamSynchronize(self.stream))?;
        }
        self.session.barrier()?;
        Ok(())
    }

    fn workspace(&mut self, bytes: usize) -> Result<Buffer, Error> {
        let elements = bytes.div_ceil(4);
        if self.working.as_ref().is_none_or(|b| b.0.len < elements) {
            self.working = Some(Device(self.context.clone()).alloc(elements)?);
            self.counters[5] += (elements * 4) as u64;
        }
        Ok(self.working.as_ref().unwrap().clone())
    }

    fn all_to_all(&mut self, input: u64, output: u64, dtype: u32,
                  sends: &[usize], receives: &[usize]) -> Result<(), Error> {
        let width = element_bytes(dtype)?;
        if sends.len() != self.world || receives.len() != self.world {
            return Err(Error("all-to-all split count must equal world size".into()));
        }
        let mut descriptor = dtype.to_le_bytes().to_vec();
        for &count in sends.iter().chain(receives) { descriptor.extend((count as u64).to_le_bytes()); }
        let rows = self.gather(descriptor)?;
        let world = self.world;
        let stride = 4 + world * 16;
        let counts = |rank: usize, receiving: bool, peer: usize| -> usize {
            let offset = rank * stride + 4 + (usize::from(receiving) * world + peer) * 8;
            u64::from_le_bytes(rows[offset..offset+8].try_into().unwrap()) as usize
        };
        if rows.len() != stride * self.world { return Err(Error("all-to-all metadata length mismatch".into())); }
        for source in 0..self.world {
            if rows[source*stride..source*stride+4] != dtype.to_le_bytes() {
                return Err(Error("all-to-all ranks disagree on dtype".into()));
            }
            for destination in 0..self.world {
                if counts(source, false, destination) != counts(destination, true, source) {
                    return Err(Error("all-to-all peer send/receive counts disagree".into()));
                }
            }
        }
        let total = |parts: &[usize]| -> Result<usize, Error> {
            parts.iter().try_fold(0usize, |a, &b| a.checked_add(b)).and_then(|n| n.checked_mul(width))
                .ok_or(Error("all-to-all byte count overflow".into()))
        };
        let (input_bytes, output_bytes) = (total(sends)?, total(receives)?);
        let valid = self.context.borrow(input, usize::from(input_bytes > 0)).is_ok()
            && self.context.borrow(output, usize::from(output_bytes > 0)).is_ok();
        if self.gather(vec![u8::from(valid)])?.contains(&0) { return Err(Error("all-to-all pointer is not on rank GPU".into())); }
        let working = if input_bytes > 0 && output_bytes > 0 && input < output.saturating_add(output_bytes as u64)
            && output < input.saturating_add(input_bytes as u64) {
            let working = self.workspace(input_bytes)?;
            self.copy(working.0.ptr, input, input_bytes, false)?;
            Some(working)
        } else { None };
        let input = working.as_ref().map_or(input, |b| b.0.ptr);
        let largest = (0..self.world).flat_map(|r| (0..self.world).map(move |p| counts(r,false,p))).max().unwrap_or(0);
        let chunk = CAPACITY / width;
        for step in 0..self.world {
            let destination = (self.rank + step) % self.world;
            let source = (self.rank + self.world - step) % self.world;
            let send_base: usize = sends[..destination].iter().sum();
            let receive_base: usize = receives[..source].iter().sum();
            for offset in (0..largest).step_by(chunk) {
                let sn = sends[destination].saturating_sub(offset).min(chunk);
                let rn = receives[source].saturating_sub(offset).min(chunk);
                self.copy(self.staging.0.ptr, input + ((send_base+offset)*width) as u64, sn*width, false)?;
                self.barrier()?;
                self.copy(output + ((receive_base+offset)*width) as u64, self.peers[source], rn*width, source != self.rank)?;
                self.barrier()?;
            }
        }
        self.barrier()?;
        self.counters[0] += 1;
        Ok(())
    }

    fn exchange_chunk(
        &mut self,
        source: u64,
        send_count: usize,
        destination: u64,
        receive_count: usize,
        dtype: u32,
        reduction: Option<u32>,
    ) -> Result<(), Error> {
        let width = element_bytes(dtype)?;
        self.copy(self.staging.0.ptr, source, send_count * width, false)?;
        self.barrier()?;
        let previous = (self.rank + self.world - 1) % self.world;
        if let Some(op) = reduction {
            self.reduce(destination, self.peers[previous], receive_count, dtype, op)?;
        } else {
            self.copy(
                destination,
                self.peers[previous],
                receive_count * width,
                true,
            )?;
        }
        self.barrier()
    }

    fn ring_reduce_scatter(
        &mut self,
        ptr: u64,
        count: usize,
        dtype: u32,
        op: u32,
    ) -> Result<(), Error> {
        let width = element_bytes(dtype)?;
        let chunk = CAPACITY / width;
        for step in 0..self.world - 1 {
            let send = (self.rank + 2 * self.world - 1 - step) % self.world;
            let receive = (self.rank + 2 * self.world - 2 - step) % self.world;
            let (send_base, send_length) = partition(count, self.world, send);
            let (recv_base, recv_length) = partition(count, self.world, receive);
            for offset in (0..count.div_ceil(self.world)).step_by(chunk) {
                let sn = send_length.saturating_sub(offset).min(chunk);
                let rn = recv_length.saturating_sub(offset).min(chunk);
                self.exchange_chunk(
                    ptr + ((send_base + offset) * width) as u64,
                    sn,
                    ptr + ((recv_base + offset) * width) as u64,
                    rn,
                    dtype,
                    Some(op),
                )?;
            }
        }
        Ok(())
    }

    fn ring_gather(&mut self, ptr: u64, count: usize, dtype: u32) -> Result<(), Error> {
        let width = element_bytes(dtype)?;
        let chunk = CAPACITY / width;
        for step in 0..self.world - 1 {
            let send = (self.rank + self.world - step) % self.world;
            let receive = (self.rank + self.world - step - 1) % self.world;
            let (send_base, send_length) = partition(count, self.world, send);
            let (recv_base, recv_length) = partition(count, self.world, receive);
            for offset in (0..count.div_ceil(self.world)).step_by(chunk) {
                let sn = send_length.saturating_sub(offset).min(chunk);
                let rn = recv_length.saturating_sub(offset).min(chunk);
                self.exchange_chunk(
                    ptr + ((send_base + offset) * width) as u64,
                    sn,
                    ptr + ((recv_base + offset) * width) as u64,
                    rn,
                    dtype,
                    None,
                )?;
            }
        }
        Ok(())
    }

    fn collective(
        &mut self,
        op: u32,
        ptr: u64,
        count: usize,
        dtype: u32,
        output: u64,
        root: u32,
    ) -> Result<(), Error> {
        let mut descriptor = Vec::new();
        descriptor.extend(op.to_le_bytes());
        descriptor.extend(dtype.to_le_bytes());
        descriptor.extend((count as u64).to_le_bytes());
        descriptor.extend(root.to_le_bytes());
        let agreements = self.gather(descriptor.clone())?;
        if agreements
            .chunks_exact(descriptor.len())
            .any(|other| other != descriptor)
        {
            return Err(Error(
                "IPC collective ranks disagree on operation/dtype/count/root".into(),
            ));
        }
        let width = element_bytes(dtype)?;
        let bytes = count
            .checked_mul(width)
            .ok_or(Error("IPC byte count overflow".into()))?;
        if !matches!(op, 0..=5)
            || root as usize >= self.world
            || (op == 5 && count % self.world != 0)
        {
            return Err(Error(
                "invalid IPC collective operation/root/scatter count".into(),
            ));
        }
        if op <= 2 && dtype > 3 {
            return Err(Error("IPC reduction supports F32/F64/I32/I64".into()));
        }
        let pointer_check = self.context.borrow(ptr, if bytes == 0 { 0 } else { 1 });
        let output_check = if matches!(op, 4 | 5) && bytes > 0 {
            self.context.borrow(output, 1).map(|_| ())
        } else {
            Ok(())
        };
        let valid = pointer_check.is_ok() && output_check.is_ok();
        if self.gather(vec![u8::from(valid)])?.contains(&0) {
            return Err(Error(
                "IPC collective input/output pointer is not on the rank GPU".into(),
            ));
        }
        match op {
            0..=2 => {
                self.ring_reduce_scatter(ptr, count, dtype, op)?;
                self.ring_gather(ptr, count, dtype)?;
            }
            3 => {
                for offset in (0..bytes).step_by(CAPACITY) {
                    let length = (bytes - offset).min(CAPACITY);
                    if self.rank == root as usize {
                        self.copy(self.staging.0.ptr, ptr + offset as u64, length, false)?;
                    }
                    self.barrier()?;
                    if self.rank != root as usize {
                        self.copy(ptr + offset as u64, self.peers[root as usize], length, true)?;
                    }
                    self.barrier()?;
                }
            }
            4 => {
                let total = count
                    .checked_mul(self.world)
                    .ok_or(Error("IPC gather count overflow".into()))?;
                total
                    .checked_mul(width)
                    .ok_or(Error("IPC gather bytes overflow".into()))?;
                self.copy(output + (self.rank * bytes) as u64, ptr, bytes, false)?;
                self.ring_gather(output, total, dtype)?;
            }
            5 => {
                if dtype > 3 {
                    return Err(Error("IPC reduction supports F32/F64/I32/I64".into()));
                }
                let working = self.workspace(bytes)?;
                self.copy(working.0.ptr, ptr, bytes, false)?;
                self.ring_reduce_scatter(working.0.ptr, count, dtype, 0)?;
                let (offset, length) = partition(count, self.world, self.rank);
                self.copy(
                    output,
                    working.0.ptr + (offset * width) as u64,
                    length * width,
                    false,
                )?;
            }
            _ => unreachable!(),
        }
        self.barrier()?;
        self.counters[0] += 1;
        Ok(())
    }
}

fn element_bytes(dtype: u32) -> Result<usize, Error> {
    match dtype {
        0 | 2 => Ok(4),
        1 | 3 => Ok(8),
        4 => Ok(1),
        5 => Ok(2),
        _ => Err(Error("unknown IPC dtype".into())),
    }
}

fn partition(count: usize, world: usize, rank: usize) -> (usize, usize) {
    (
        rank * (count / world) + rank.min(count % world),
        count / world + usize::from(rank < count % world),
    )
}

impl Drop for IpcGroup {
    fn drop(&mut self) {
        let _ = self.context.bind();
        if !self.stream.is_null() { unsafe { let _ = cuStreamSynchronize(self.stream); } }
        for (rank, &ptr) in self.peers.iter().enumerate() {
            if rank != self.rank && ptr != 0 {
                unsafe {
                    let _ = cuIpcCloseMemHandle(ptr);
                }
            }
        }
        if !self.module.is_null() {
            unsafe {
                let _ = cuModuleUnload(self.module);
            }
        }
        if !self.stream.is_null() {
            unsafe { let _ = cuStreamSynchronize(self.stream); let _ = cuStreamDestroy_v2(self.stream); }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruccl_ipc_api_version() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ipc_create(
    rank: u32,
    world: u32,
    device: i32,
    port: u32,
    uid: *const u8,
    timeout_ms: u64,
    out: *mut *mut c_void,
) -> i32 {
    checked(|| {
        if uid.is_null()
            || out.is_null()
            || world == 0
            || rank >= world
            || port > u16::MAX as u32
            || timeout_ms == 0
        {
            return Err(Error("invalid IPC communicator configuration".into()));
        }
        let id = UniqueId::from_bytes(
            unsafe { std::slice::from_raw_parts(uid, 16) }
                .try_into()
                .unwrap(),
        );
        let session = Arc::new(TcpRankSession::connect(
            ("127.0.0.1", port as u16),
            id,
            rank,
            world,
            Duration::from_millis(timeout_ms),
        )?);
        let context = Context::new(device)?;
        let staging = Device(context.clone()).alloc(CAPACITY / 4)?;
        let mut group = IpcGroup {
            context,
            session,
            rank: rank as usize,
            world: world as usize,
            staging,
            peers: vec![0; world as usize],
            module: std::ptr::null_mut(),
            reductions: [std::ptr::null_mut(); 4],
            stream: std::ptr::null_mut(),
            working: None,
            counters: [0, 0, 0, 0, 0, CAPACITY as u64, 0, 0],
        };
        unsafe { check(cuStreamCreate(&mut group.stream, CUstream_flags::CU_STREAM_NON_BLOCKING as u32))?; }
        let mut handle = CUipcMemHandle { reserved: [0; 64] };
        let export = if world > 1 {
            unsafe { cuIpcGetMemHandle(&mut handle, group.staging.0.ptr) }
        } else {
            CUresult::CUDA_SUCCESS
        };
        let mut payload = vec![u8::from(export == CUresult::CUDA_SUCCESS)];
        payload.extend(handle.reserved.iter().map(|&byte| byte as u8));
        let descriptors = group.gather(payload)?;
        if descriptors.chunks_exact(65).any(|peer| peer[0] == 0) {
            return Err(Error(
                "CUDA IPC export unavailable; a Linux CUDA IPC-capable environment is required"
                    .into(),
            ));
        }
        let mut import_error = None;
        for (peer, descriptor) in descriptors.chunks_exact(65).enumerate() {
            if peer == rank as usize {
                group.peers[peer] = group.staging.0.ptr;
                continue;
            }
            let handle = CUipcMemHandle {
                reserved: std::array::from_fn(|i| descriptor[i + 1] as i8),
            };
            let result = unsafe { cuIpcOpenMemHandle_v2(&mut group.peers[peer], handle, 1) };
            if result != CUresult::CUDA_SUCCESS {
                import_error = Some(format!("CUDA IPC peer {peer}: {result:?}"));
            }
        }
        if group
            .gather(vec![u8::from(import_error.is_none())])?
            .contains(&0)
        {
            return Err(Error(import_error.unwrap_or_else(|| {
                "another rank failed to import CUDA IPC peer memory".into()
            })));
        }
        let ptx = CString::new(reduction_ptx()).unwrap();
        unsafe {
            check(cuModuleLoadData(&mut group.module, ptx.as_ptr().cast()))?;
        }
        for i in 0..4 {
            let name = CString::new(format!("reduce_{i}")).unwrap();
            unsafe {
                check(cuModuleGetFunction(
                    &mut group.reductions[i],
                    group.module,
                    name.as_ptr(),
                ))?;
            }
        }
        group.barrier()?;
        unsafe {
            *out = Box::into_raw(Box::new(IpcHandle {
                session: group.session.clone(), context: group.context.clone(), rank, world,
                group: Mutex::new(group), p2p_counters: std::array::from_fn(|_| AtomicU64::new(0)),
                p2p_send_buffers: Mutex::new(std::collections::HashMap::new()),
            })).cast();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ipc_collective(
    handle: *mut c_void,
    op: u32,
    ptr: u64,
    count: usize,
    dtype: u32,
    output: u64,
    root: u32,
) -> i32 {
    checked(|| {
        if handle.is_null() {
            return Err(Error("null IPC communicator".into()));
        }
        let handle = unsafe { &*handle.cast::<IpcHandle>() };
        let mut group = handle.group.lock().map_err(|e| Error(e.to_string()))?;
        let result = group.collective(op, ptr, count, dtype, output, root);
        if let Err(error) = &result {
            let _ = group.session.abort(&error.to_string());
        }
        result
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ipc_stats(handle: *const c_void, out: *mut u64) -> i32 {
    checked(|| {
        if handle.is_null() || out.is_null() {
            return Err(Error("null IPC stats pointer".into()));
        }
        let handle = unsafe { &*handle.cast::<IpcHandle>() };
        let group = handle.group.lock().map_err(|e| Error(e.to_string()))?;
        let values: [u64; 8] = std::array::from_fn(|i| group.counters[i] + handle.p2p_counters[i].load(Ordering::Relaxed));
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, 8);
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ipc_all_to_all(
    handle: *mut c_void, input: u64, output: u64, dtype: u32,
    sends: *const usize, receives: *const usize,
) -> i32 {
    checked(|| {
        if handle.is_null() || sends.is_null() || receives.is_null() { return Err(Error("null all-to-all argument".into())); }
        let handle = unsafe { &*handle.cast::<IpcHandle>() };
        let sends = unsafe { std::slice::from_raw_parts(sends, handle.world as usize) };
        let receives = unsafe { std::slice::from_raw_parts(receives, handle.world as usize) };
        let mut group = handle.group.lock().map_err(|e| Error(e.to_string()))?;
        let result = group.all_to_all(input, output, dtype, sends, receives);
        if let Err(error) = &result { let _ = handle.session.abort(&error.to_string()); }
        result
    })
}

impl IpcHandle {
    fn transfer(&self, receive: bool, peer: u32, tag: u64, ptr: u64, count: usize, dtype: u32) -> Result<(), Error> {
        if peer >= self.world || peer == self.rank || tag > (u64::MAX - 1) / 2 {
            return Err(Error("invalid IPC point-to-point peer/tag".into()));
        }
        let width = element_bytes(dtype)?;
        let bytes = count.checked_mul(width).ok_or(Error("IPC point-to-point byte count overflow".into()))?;
        let _borrowed = self.context.borrow(ptr, usize::from(bytes > 0))?;
        self.context.bind()?;
        let stream = TransferStream::new()?;
        let message_tag = tag * 2;
        if receive {
            let descriptor = self.session.receive(Some(peer), message_tag, ElementType::U8, 76)?.payload;
            if descriptor.len() != 76 || u64::from_le_bytes(descriptor[..8].try_into().unwrap()) != count as u64
                || u32::from_le_bytes(descriptor[8..12].try_into().unwrap()) != dtype {
                return Err(Error("IPC point-to-point count/dtype mismatch".into()));
            }
            let handle = CUipcMemHandle { reserved: std::array::from_fn(|i| descriptor[i+12] as i8) };
            let mut source = 0;
            unsafe { check(cuIpcOpenMemHandle_v2(&mut source, handle, 1))?; }
            let copied = (|| {
                if bytes > 0 { unsafe { check(cuMemcpyDtoDAsync_v2(ptr, source, bytes, stream.0))?; } }
                stream.synchronize()
            })();
            unsafe { check(cuIpcCloseMemHandle(source))?; }
            copied?;
            self.session.send(peer, message_tag+1, ElementType::U8, 1, vec![1])?;
            self.p2p_counters[1].fetch_add(bytes as u64, Ordering::Relaxed);
        } else {
            let elements = bytes.div_ceil(4).max(1);
            let buffer_slot = self.p2p_send_buffers.lock().map_err(|e| Error(e.to_string()))?
                .entry(peer).or_insert_with(|| Arc::new(Mutex::new(None))).clone();
            let mut buffer_guard = buffer_slot.lock().map_err(|e| Error(e.to_string()))?;
            let storage = {
                if buffer_guard.as_ref().is_none_or(|b| b.0.len < elements) {
                    *buffer_guard = Some(Device(self.context.clone()).alloc(elements)?);
                    self.p2p_counters[5].fetch_add((elements*4) as u64, Ordering::Relaxed);
                }
                buffer_guard.as_ref().unwrap().clone()
            };
            if bytes > 0 { unsafe { check(cuMemcpyDtoDAsync_v2(storage.0.ptr, ptr, bytes, stream.0))?; } }
            stream.synchronize()?;
            let mut handle = CUipcMemHandle { reserved: [0; 64] };
            unsafe { check(cuIpcGetMemHandle(&mut handle, storage.0.ptr))?; }
            let mut descriptor = (count as u64).to_le_bytes().to_vec();
            descriptor.extend(dtype.to_le_bytes());
            descriptor.extend(handle.reserved.iter().map(|&b| b as u8));
            self.session.send(peer, message_tag, ElementType::U8, 76, descriptor)?;
            let ack = self.session.receive(Some(peer), message_tag+1, ElementType::U8, 1)?;
            if ack.payload != [1] { return Err(Error("invalid IPC point-to-point acknowledgement".into())); }
        }
        self.p2p_counters[0].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ipc_p2p(
    handle: *mut c_void, receive: u32, peer: u32, tag: u64, ptr: u64, count: usize, dtype: u32,
) -> i32 {
    checked(|| {
        if handle.is_null() || receive > 1 { return Err(Error("invalid IPC point-to-point handle/direction".into())); }
        let handle = unsafe { &*handle.cast::<IpcHandle>() };
        let result = handle.transfer(receive != 0, peer, tag, ptr, count, dtype);
        if let Err(error) = &result { let _ = handle.session.abort(&error.to_string()); }
        result
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ipc_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle.cast::<IpcHandle>()) });
    }
}
