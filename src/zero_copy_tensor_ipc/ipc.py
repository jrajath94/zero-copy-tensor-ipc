import numpy as np
from multiprocessing import shared_memory
from typing import Tuple, Optional

class SharedMemoryTensor:
    def __init__(self, name: str, shape: Tuple, dtype: str, mode: str = "r"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        
    def attach(self) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=self.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return arr.copy()

def create_shared_tensor(name: str, data: np.ndarray) -> SharedMemoryTensor:
    """Create shared memory tensor."""
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    return SharedMemoryTensor(shm.name, data.shape, str(data.dtype))

def attach_shared_tensor(name: str, shape: Tuple, dtype: str) -> np.ndarray:
    """Attach to existing shared memory tensor."""
    shm = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)
