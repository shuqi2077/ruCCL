use super::*;
use std::ffi::{c_char,CStr};
use std::sync::Mutex;

#[repr(C)]
pub struct Transfer {peer:u32,receive:u32,tag:u64,pointer:u64,bytes:usize}
unsafe extern "C" {
    fn ruccl_ucx_c_error()->*const c_char;
    fn ruccl_ucx_c_create(world:u32,transports:*const c_char,out:*mut *mut c_void)->i32;
    fn ruccl_ucx_c_address(handle:*mut c_void,destination:*mut u8,length:*mut usize)->i32;
    fn ruccl_ucx_c_connect(handle:*mut c_void,peer:u32,address:*const u8)->i32;
    fn ruccl_ucx_c_exchange(handle:*mut c_void,transfers:*const Transfer,count:usize,timeout_ms:u64)->i32;
    fn ruccl_ucx_c_destroy(handle:*mut c_void);
}
fn status(code:i32)->Result<(),Error> {
    if code==0 {Ok(())} else {Err(Error(unsafe {CStr::from_ptr(ruccl_ucx_c_error())}.to_string_lossy().into_owned()))}
}
struct UcxHandle {
    context:Arc<Context>,worker:*mut c_void,module:CUmodule,functions:[CUfunction;4],
    reduction:Mutex<CUstream>,
}
unsafe impl Send for UcxHandle {}
unsafe impl Sync for UcxHandle {}
impl Drop for UcxHandle {
    fn drop(&mut self) {let _=self.context.bind();unsafe {
        if !self.worker.is_null() {ruccl_ucx_c_destroy(self.worker);}
        if let Ok(stream)=self.reduction.get_mut() {
            if !stream.is_null() {let _=cuStreamSynchronize(*stream);let _=cuStreamDestroy_v2(*stream);}
        }
        if !self.module.is_null() {let _=cuModuleUnload(self.module);}
    }}
}
unsafe fn handle<'a>(raw:*mut c_void)->Result<&'a UcxHandle,Error> {
    if raw.is_null() {return Err(Error("null UCX handle".into()));}
    let h=unsafe {&*raw.cast::<UcxHandle>()};h.context.bind()?;Ok(h)
}
#[unsafe(no_mangle)]
pub extern "C" fn ruccl_ucx_api_version()->u32 {1}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ucx_create(world:u32,device:i32,transports:*const c_char,out:*mut *mut c_void)->i32 {
    checked(|| {
        if out.is_null() || transports.is_null() {return Err(Error("null UCX arguments".into()));}
        let context=Context::new(device)?;
        let mut h=Box::new(UcxHandle{context,worker:std::ptr::null_mut(),module:std::ptr::null_mut(),
            functions:[std::ptr::null_mut();4],reduction:Mutex::new(std::ptr::null_mut())});
        unsafe {
            status(ruccl_ucx_c_create(world,transports,&mut h.worker))?;
            let ptx=CString::new(ipc::reduction_ptx()).unwrap();check(cuModuleLoadData(&mut h.module,ptx.as_ptr().cast()))?;
            for i in 0..4 {let name=CString::new(format!("reduce_{i}")).unwrap();
                check(cuModuleGetFunction(&mut h.functions[i],h.module,name.as_ptr()))?;}
            check(cuStreamCreate(h.reduction.get_mut().unwrap(),CUstream_flags::CU_STREAM_NON_BLOCKING as u32))?;
            *out=Box::into_raw(h).cast();
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ucx_address(raw:*mut c_void,destination:*mut u8,length:*mut usize)->i32 {
    checked(|| {if length.is_null() {return Err(Error("null UCX address length".into()));}
        let h=unsafe {handle(raw)?};status(unsafe {ruccl_ucx_c_address(h.worker,destination,length)})})
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ucx_connect(raw:*mut c_void,peer:u32,address:*const u8)->i32 {
    checked(|| {let h=unsafe {handle(raw)?};status(unsafe {ruccl_ucx_c_connect(h.worker,peer,address)})})
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ucx_exchange(raw:*mut c_void,transfers:*const Transfer,count:usize,timeout_ms:u64)->i32 {
    checked(|| {let h=unsafe {handle(raw)?};
        if count>0 && transfers.is_null() {return Err(Error("null UCX descriptors".into()));}
        if count>0 {for t in unsafe {std::slice::from_raw_parts(transfers,count)} {let _=h.context.borrow(t.pointer,t.bytes)?;}}
        status(unsafe {ruccl_ucx_c_exchange(h.worker,transfers,count,timeout_ms)})})
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ucx_reduce(raw:*mut c_void,destination:u64,source:u64,count:usize,dtype:u32,op:u32)->i32 {
    checked(|| {
        let h=unsafe {handle(raw)?};
        if dtype>3 || op>2 {return Err(Error("UCX reduction requires F32/F64/I32/I64 and SUM/MAX/MIN".into()));}
        if count==0 {return Ok(());}
        let _=h.context.borrow(destination,count)?;let _=h.context.borrow(source,count)?;
        let mut n=u32::try_from(count).map_err(|e|Error(e.to_string()))?;
        let (mut a,mut b,mut operation)=(source,destination,op);
        let mut args=[(&mut a as *mut u64).cast::<c_void>(),(&mut b as *mut u64).cast(),
            (&mut n as *mut u32).cast(),(&mut operation as *mut u32).cast()];
        let stream=h.reduction.lock().map_err(|_|Error("UCX reduction stream poisoned".into()))?;
        unsafe {check(cuLaunchKernel(h.functions[dtype as usize],n.div_ceil(256),1,1,256,1,1,0,*stream,
            args.as_mut_ptr(),std::ptr::null_mut()))?;check(cuStreamSynchronize(*stream))?;}
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_ucx_destroy(raw:*mut c_void) {
    if !raw.is_null() {drop(unsafe {Box::from_raw(raw.cast::<UcxHandle>())});}
}
