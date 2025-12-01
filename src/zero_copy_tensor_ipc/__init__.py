"""Zero-copy tensor sharing via shared memory IPC."""
from .ipc import SharedMemoryTensor, create_shared_tensor, attach_shared_tensor

__version__ = "1.0.0"
__all__ = ["SharedMemoryTensor", "create_shared_tensor", "attach_shared_tensor"]
