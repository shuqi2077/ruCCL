"""Owned GPU buffers; reductions execute on the explicitly selected RUDA backend."""
from array import array
import ctypes
import math
import os
import threading

BACKENDS = {'rocm': 2, 'metal': 3, 'vulkan': 4}
DEVICE_CLASSES = {'discrete': 0, 'integrated': 1, 'virtual': 2, 'default': 4}


def library(path=None):
    path = path or os.environ['RUCCL_TORCH_LIBRARY']
    dependency_directory = (os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
                            if os.name == 'nt' else None)
    try:
        native = ctypes.CDLL(path)
    except BaseException:
        if dependency_directory is not None:
            dependency_directory.close()
        raise
    native._dependency_directory = dependency_directory
    native.ruccl_error.restype = ctypes.c_char_p
    native.ruccl_bridge_abi_version.restype = ctypes.c_uint32
    if native.ruccl_bridge_abi_version() != 1:
        raise RuntimeError('ruCCL bridge ABI mismatch')
    native.ruccl_backend_mask.restype = ctypes.c_uint32
    native.ruccl_server.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    native.ruccl_tensor_create.argtypes = [ctypes.c_uint32]*6 + [ctypes.c_void_p,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    native.ruccl_tensor_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    native.ruccl_tensor_download.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_size_t]
    native.ruccl_tensor_collective.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    native.ruccl_tensor_release.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    native.ruccl_tensor_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    native.ruccl_tensor_destroy.argtypes = [ctypes.c_void_p]
    native.ruccl_tensor_destroy.restype = None
    return native


def check(native, status):
    if status:
        raise RuntimeError(native.ruccl_error().decode('utf-8', errors='replace'))


def rendezvous(world_size, *, library_path=None):
    """Return (16-byte session ID, localhost TCP port). Call once per group."""
    native = library(library_path)
    uid = (ctypes.c_uint8*16)()
    port = ctypes.c_uint32()
    check(native, native.ruccl_server(world_size, uid, ctypes.byref(port)))
    return bytes(uid), port.value


class Buffer:
    dtype = 'float32'

    def __init__(self, owner, identifier, shape):
        self.owner, self.identifier, self.shape = owner, identifier, tuple(shape)

    def download(self):
        values = array('f', [0])*math.prod(self.shape)
        self.owner.download_address(self, values.buffer_info()[0])
        return values

    def close(self):
        with self.owner.lock:
            if self.identifier is not None:
                if self.owner.handle.value:
                    check(self.owner.native, self.owner.native.ruccl_tensor_release(
                        self.owner.handle, self.identifier))
                self.identifier = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Communicator:
    def __init__(self, rank, world_size, session, *, backend, device=0,
                 device_class='default', algorithm='auto', library_path=None):
        if backend not in BACKENDS or device_class not in DEVICE_CLASSES:
            raise ValueError('select rocm/metal/vulkan and a GPU device class')
        if algorithm not in ('auto', 'ring'):
            raise ValueError('algorithm must be auto or ring')
        if not isinstance(device, int) or device < 0:
            raise ValueError('device must be a nonnegative ordinal')
        self.native = library(library_path)
        if not self.native.ruccl_backend_mask() & (1 << (BACKENDS[backend]-1)):
            raise RuntimeError('backend not compiled for this platform')
        self.rank, self.world_size, self.backend = rank, world_size, backend
        self.handle = ctypes.c_void_p()
        self.lock = threading.RLock()
        uid_bytes, port = session
        uid = (ctypes.c_uint8*16).from_buffer_copy(uid_bytes)
        check(self.native, self.native.ruccl_tensor_create(rank, world_size, BACKENDS[backend],
            DEVICE_CLASSES[device_class], device, port, uid, int(algorithm == 'ring'), ctypes.byref(self.handle)))

    def _validate(self, buffer=None):
        if not self.handle.value:
            raise RuntimeError('communicator is closed')
        if buffer is not None and (buffer.owner is not self or buffer.identifier is None):
            raise ValueError('buffer belongs to another communicator or was released')

    def upload(self, values, shape=None):
        values = array('f', values)
        shape = (len(values),) if shape is None else tuple(shape)
        if math.prod(shape) != len(values):
            raise ValueError('shape does not match values')
        return self.upload_address(values.buffer_info()[0], shape)

    def upload_address(self, address, shape):
        """Borrow an in-process FP32 host address for this synchronous call."""
        shape = tuple(shape)
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError('invalid shape')
        with self.lock:
            self._validate()
            output = ctypes.c_uint64()
            check(self.native, self.native.ruccl_tensor_upload(
                self.handle, address, math.prod(shape), ctypes.byref(output)))
            return Buffer(self, output.value, shape)

    def download_address(self, buffer, address):
        """Borrow a writable in-process FP32 host address for this synchronous call."""
        with self.lock:
            self._validate(buffer)
            check(self.native, self.native.ruccl_tensor_download(
                self.handle, buffer.identifier, address, math.prod(buffer.shape)))

    def collective(self, buffer, operation, *, root=0):
        if operation not in range(6) or not 0 <= root < self.world_size:
            raise ValueError('invalid operation or root')
        shape = list(buffer.shape)
        if operation in (4, 5):
            if not shape or (operation == 5 and shape[0] % self.world_size):
                raise ValueError('gather/scatter requires a valid axis-zero shard')
            shape[0] = shape[0]*self.world_size if operation == 4 else shape[0]//self.world_size
        with self.lock:
            self._validate(buffer)
            output = ctypes.c_uint64()
            check(self.native, self.native.ruccl_tensor_collective(
                self.handle, buffer.identifier, operation, root, ctypes.byref(output)))
            return buffer if output.value == buffer.identifier else Buffer(self, output.value, shape)

    def stats(self):
        with self.lock:
            self._validate()
            values = (ctypes.c_uint64*4)()
            check(self.native, self.native.ruccl_tensor_stats(self.handle, values))
            return dict(calls=values[0], protocol_bytes=values[1], reduction_kernel_launches=values[2],
                        wall_seconds=values[3]/1e9, provider='ruCCL', backend=self.backend, transport='TcpHostStaged')

    def close(self):
        with self.lock:
            if self.handle.value:
                self.native.ruccl_tensor_destroy(self.handle)
                self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
