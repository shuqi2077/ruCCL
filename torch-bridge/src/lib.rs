use ruccl::rank::{NetworkError, TopologyError, UniqueId, TcpRendezvousServer,
    error::RankError, work::WorkError};
use std::{cell::RefCell, ffi::{CString, c_char}, fmt,
    panic::{catch_unwind, AssertUnwindSafe}};

#[cfg(feature = "cuda")]
mod cuda;
mod tensor;

#[derive(Debug)]
pub(crate) struct Error(pub(crate) String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
}
impl std::error::Error for Error {}
macro_rules! from_error {
    ($($t:ty),*) => {$(impl From<$t> for Error {
        fn from(e: $t) -> Self { Self(e.to_string()) }
    })*};
}
from_error!(RankError, NetworkError, TopologyError, WorkError,
    ruccl::tensor_device::TensorDeviceError);
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }
pub(crate) fn checked(f: impl FnOnce() -> Result<(), Error>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        failure => {
            let message = match failure {
                Ok(Err(error)) => error.to_string(),
                Err(error) => error.downcast_ref::<String>().cloned()
                    .or_else(|| error.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "ruCCL panic".into()),
                _ => unreachable!(),
            };
            ERROR.with(|slot| *slot.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            -1
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn ruccl_error() -> *const c_char { ERROR.with(|slot| slot.borrow().as_ptr()) }

#[unsafe(no_mangle)]
pub extern "C" fn ruccl_bridge_abi_version() -> u32 { 1 }

// Bits: CUDA raw-pointer bridge, ROCm tensor bridge, Metal, Vulkan.
#[unsafe(no_mangle)]
pub extern "C" fn ruccl_backend_mask() -> u32 {
    u32::from(cfg!(feature = "cuda")) | (u32::from(cfg!(feature = "rocm")) << 1)
        | (u32::from(cfg!(all(feature = "wgpu", target_os = "macos"))) << 2)
        | (u32::from(cfg!(feature = "wgpu")) << 3)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruccl_server(world: u32, uid: *mut u8, port: *mut u32) -> i32 {
    checked(|| {
        if uid.is_null() || port.is_null() { return Err(Error("null rendezvous output".into())); }
        let id = UniqueId::new();
        let server = TcpRendezvousServer::bind("127.0.0.1:0", id, world as usize)?;
        unsafe {
            std::ptr::copy_nonoverlapping(id.as_bytes().as_ptr(), uid, 16);
            *port = server.local_addr()?.port() as u32;
        }
        std::thread::spawn(move || {
            if let Err(error) = server.run() { eprintln!("ruCCL rendezvous: {error}"); }
        });
        Ok(())
    })
}
