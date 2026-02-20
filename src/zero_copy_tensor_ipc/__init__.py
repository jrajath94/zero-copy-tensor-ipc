"""Zero-copy tensor sharing via shared memory IPC."""
from .ipc import (
    SharedMemoryTensor,
    TensorMetadata,
    create_shared_tensor,
    attach_shared_tensor,
    list_active_segments,
    cleanup_all_segments,
)

__version__ = "1.0.0"
__all__ = [
    "SharedMemoryTensor",
    "TensorMetadata",
    "create_shared_tensor",
    "attach_shared_tensor",
    "list_active_segments",
    "cleanup_all_segments",
]
