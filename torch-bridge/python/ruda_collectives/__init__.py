"""Communication helpers without initializing the CUDA-only RUDA compute extension."""
from .distributed_training import ReplicaGroup
from torch.distributed import is_available as _distributed_available
if _distributed_available():
    from .process_group import ProcessGroupRuCCL, register_ruccl_backend
