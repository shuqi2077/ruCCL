use cudarc::driver::sys::*;
use crate::{Error, checked};
use ruccl::rank::{AlgorithmPolicy, CollectiveTuning, ElementType,
    ReductionOperation, TcpRankSession, UniqueId,
    communicator::RankCommunicator, device::RankDevice};
use std::{ffi::{CString,c_void},
    sync::{Arc,atomic::{AtomicU64,Ordering}},time::{Duration,Instant}};

mod ipc;

fn check(s:CUresult)->Result<(),Error> {if s==CUresult::CUDA_SUCCESS {Ok(())}else{Err(Error(format!("CUDA {s:?}")))}}
const PTX:&str=r#".version 6.0
.target sm_50
.address_size 64
.visible .entry reduce(.param .u64 src,.param .u64 dst,.param .u32 n,.param .u32 off,.param .u32 op) {
 .reg .pred p,q; .reg .b32 i,t,s,l,o,k; .reg .b64 a,b,x,y; .reg .f32 v,w,z;
 ld.param.u64 a,[src]; ld.param.u64 b,[dst]; ld.param.u32 l,[n]; ld.param.u32 o,[off]; ld.param.u32 k,[op];
 mov.u32 i,%ctaid.x; mov.u32 t,%ntid.x; mov.u32 s,%tid.x; mad.lo.u32 i,i,t,s;
 setp.ge.u32 p,i,l; @p bra END;
 mul.wide.u32 x,i,4; add.u64 x,a,x; add.u32 t,i,o; mul.wide.u32 y,t,4; add.u64 y,b,y;
 ld.global.f32 v,[x]; ld.global.f32 w,[y];
 setp.eq.u32 q,k,0; @q bra SUM; setp.eq.u32 q,k,1; @q bra MAX;
 min.f32 z,v,w; bra STORE;
 SUM: add.f32 z,v,w; bra STORE;
 MAX: max.f32 z,v,w;
 STORE: st.global.f32 [y],z;
 END: ret;
}"#;
struct Context {device:i32,ctx:CUcontext,module:CUmodule,function:CUfunction,counters:[AtomicU64;5]}
unsafe impl Send for Context {} unsafe impl Sync for Context {}
impl Context {
    fn new(device:i32)->Result<Arc<Self>,Error> {unsafe {
        check(cuInit(0))?;let mut ctx=std::ptr::null_mut();check(cuDevicePrimaryCtxRetain(&mut ctx,device))?;
        check(cuCtxSetCurrent(ctx))?;let mut module=std::ptr::null_mut();let ptx=CString::new(PTX).unwrap();
        check(cuModuleLoadData(&mut module,ptx.as_ptr().cast()))?;
        let mut function=std::ptr::null_mut();check(cuModuleGetFunction(&mut function,module,c"reduce".as_ptr()))?;
        Ok(Arc::new(Self{device,ctx,module,function,counters:std::array::from_fn(|_|AtomicU64::new(0))}))
    }}
    fn bind(&self)->Result<(),Error> {unsafe{check(cuCtxSetCurrent(self.ctx))}}
    fn borrow(self:&Arc<Self>,ptr:u64,len:usize)->Result<Buffer,Error> {
        self.bind()?;let mut device=0;
        if len>0 {unsafe {check(cuPointerGetAttribute((&mut device as *mut i32).cast(),CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,ptr))?;}}
        if len>0 && device!=self.device {return Err(Error("pointer device mismatch".into()));}
        Ok(Buffer(Arc::new(Storage{context:self.clone(),ptr,len,owned:false})))
    }
}
impl Drop for Context {fn drop(&mut self){unsafe{let _=cuCtxSetCurrent(self.ctx);let _=cuModuleUnload(self.module);let _=cuDevicePrimaryCtxRelease_v2(self.device);}}}
struct Storage {context:Arc<Context>,ptr:u64,len:usize,owned:bool}
impl Drop for Storage {fn drop(&mut self){if self.owned {let _=self.context.bind();unsafe{let _=cuMemFree_v2(self.ptr);}}}}
#[derive(Clone)] struct Buffer(Arc<Storage>);
#[derive(Clone)] struct Device(Arc<Context>);
impl RankDevice<f32> for Device {
    type Buffer=Buffer;type Kernel=ReductionOperation;type ReductionLaunch=(u32,u32);type Error=Error;
    const ELEMENT_SIZE:usize=4;
    fn buffer_len(&self,b:&Buffer)->usize{b.0.len}
    fn buffer_bytes(&self,b:&Buffer)->usize{b.0.len*4}
    fn encode(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
    fn decode(v:&[u8])->Result<Vec<f32>,Error>{if v.len()%4!=0{return Err(Error("unaligned FP32".into()));}Ok(v.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect())}
    fn alloc(&self,n:usize)->Result<Buffer,Error>{self.0.bind()?;let mut ptr=0;if n>0{unsafe{check(cuMemAlloc_v2(&mut ptr,n.checked_mul(4).ok_or(Error("allocation overflow".into()))?))?;}}
        self.0.counters[2].fetch_add((n*4)as u64,Ordering::Relaxed);Ok(Buffer(Arc::new(Storage{context:self.0.clone(),ptr,len:n,owned:true})))}
    fn copy_to_device(&self,b:&Buffer,v:&[f32])->Result<(),Error>{self.copy_bytes_to_device_at(b,0,&Self::encode(v))}
    fn copy_from_device(&self,b:&Buffer)->Result<Vec<f32>,Error>{self.copy_from_device_at(b,0,b.0.len)}
    fn copy_from_device_at(&self,b:&Buffer,o:usize,n:usize)->Result<Vec<f32>,Error>{Self::decode(&self.copy_bytes_from_device_at(b,o,n)?)}
    fn copy_bytes_to_device_at(&self,b:&Buffer,o:usize,v:&[u8])->Result<(),Error>{
        if v.len()%4!=0 || o.checked_add(v.len()/4).is_none_or(|e|e>b.0.len){return Err(Error("write range".into()));}
        self.0.bind()?;let start=Instant::now();if !v.is_empty(){unsafe{check(cuMemcpyHtoD_v2(b.0.ptr+(o*4)as u64,v.as_ptr().cast(),v.len()))?;}}
        self.0.counters[1].fetch_add(v.len()as u64,Ordering::Relaxed);self.0.counters[4].fetch_add(start.elapsed().as_nanos()as u64,Ordering::Relaxed);Ok(())}
    fn copy_bytes_from_device_at(&self,b:&Buffer,o:usize,n:usize)->Result<Vec<u8>,Error>{
        if o.checked_add(n).is_none_or(|e|e>b.0.len){return Err(Error("read range".into()));}
        self.0.bind()?;let mut out=vec![0;n*4];let start=Instant::now();if n>0{unsafe{check(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(),b.0.ptr+(o*4)as u64,n*4))?;}}
        self.0.counters[0].fetch_add((n*4)as u64,Ordering::Relaxed);self.0.counters[3].fetch_add(start.elapsed().as_nanos()as u64,Ordering::Relaxed);Ok(out)}
    fn prepare_reduction(&self,n:usize,o:usize)->Result<(u32,u32),Error>{Ok((u32::try_from(n).map_err(|e|Error(e.to_string()))?,u32::try_from(o).map_err(|e|Error(e.to_string()))?))}
    fn launch_reduction(&self,k:&ReductionOperation,l:&(u32,u32),src:&Buffer,dst:&Buffer)->Result<(),Error>{
        if l.0==0{return Ok(());}if l.0 as usize>src.0.len || l.0 as usize+l.1 as usize>dst.0.len{return Err(Error("reduction range".into()));}
        let mut op=match k{ReductionOperation::Sum=>0_u32,ReductionOperation::Maximum=>1,ReductionOperation::Minimum=>2,_=>return Err(Error("unsupported reduction".into()))};
        self.0.bind()?;let(mut a,mut b,mut n,mut o)=(src.0.ptr,dst.0.ptr,l.0,l.1);
        let mut args=[(&mut a as *mut u64).cast::<c_void>(),(&mut b as *mut u64).cast(),(&mut n as *mut u32).cast(),(&mut o as *mut u32).cast(),(&mut op as *mut u32).cast()];
        unsafe{check(cuLaunchKernel(self.0.function,n.div_ceil(256),1,1,256,1,1,0,std::ptr::null_mut(),args.as_mut_ptr(),std::ptr::null_mut()))?;check(cuCtxSynchronize())?;}Ok(())}
}
struct Group{comm:RankCommunicator<Device>,counters:[u64;3]}
#[unsafe(no_mangle)] pub unsafe extern "C" fn ruccl_create(rank:u32,world:u32,device:i32,port:u32,uid:*const u8,ring:u32,out:*mut *mut c_void)->i32{
    checked(||{let id=UniqueId::from_bytes(unsafe{std::slice::from_raw_parts(uid,16)}.try_into().unwrap());
        let session=TcpRankSession::connect(("127.0.0.1",port as u16),id,rank,world,Duration::from_secs(300))?;
        let mut tuning=CollectiveTuning::from_environment(world)?;if ring!=0{tuning.algorithm_policy=AlgorithmPolicy::Ring;}
        let comm=RankCommunicator::from_session_with_tuning(||Context::new(device).map(Device),session,tuning,"ruccl-torch")?;
        unsafe{*out=Box::into_raw(Box::new(Group{comm,counters:[0;3]})).cast();}Ok(())})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn ruccl_collective(handle:*mut c_void,op:u32,ptr:u64,n:usize,output:u64,root:u32)->i32{
    checked(||{let g=unsafe{&mut *handle.cast::<Group>()};let d=g.comm.execution();let b=d.0.borrow(ptr,n)?;
        let c=g.comm.device_collective::<f32>(ElementType::F32);let stats=match op{
            0|1|2=>{let k=match op{0=>ReductionOperation::Sum,1=>ReductionOperation::Maximum,_=>ReductionOperation::Minimum};c.all_reduce(&b,k,&k)?},
            3=>c.broadcast(&b,root)?,
            4|5=>{let(v,s)=if op==4{c.all_gather(&b)?}else{c.reduce_scatter(&b,ReductionOperation::Sum,&ReductionOperation::Sum)?};
                let _=d.0.borrow(output,v.0.len)?;d.0.bind()?;if v.0.len>0{unsafe{check(cuMemcpyDtoD_v2(output,v.0.ptr,v.0.len*4))?;check(cuCtxSynchronize())?;}}s},
            _=>return Err(Error("unknown collective".into()))};
        g.counters[0]+=1;g.counters[1]+=stats.transferred_bytes as u64;g.counters[2]+=stats.reduction_kernel_launches as u64;Ok(())})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn ruccl_stats(handle:*const c_void,out:*mut u64)->i32{
    checked(||{let g=unsafe{&*handle.cast::<Group>()};let mut values=g.counters.to_vec();values.extend(g.comm.execution().0.counters.iter().map(|c|c.load(Ordering::Relaxed)));
        unsafe{std::ptr::copy_nonoverlapping(values.as_ptr(),out,8);}Ok(())})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn ruccl_destroy(handle:*mut c_void){if !handle.is_null(){drop(unsafe{Box::from_raw(handle.cast::<Group>())});}}
