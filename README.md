# zero-copy-tensor-ipc

> Move gigabyte tensors between Python processes in microseconds instead of milliseconds

[![CI](https://github.com/jrajath94/zero-copy-tensor-ipc/workflows/CI/badge.svg)](https://github.com/jrajath94/zero-copy-tensor-ipc/actions)
[![Coverage](https://codecov.io/gh/jrajath94/zero-copy-tensor-ipc/branch/master/graph/badge.svg)](https://codecov.io/gh/jrajath94/zero-copy-tensor-ipc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Why This Exists

Python's multiprocessing module was designed to work around the GIL. You spawn separate processes to get true parallelism. But then you need to move data between those processes, and Python's default mechanism serializes everything into a byte stream, sends it through a pipe, and deserializes on the other side. For small objects, this is fine. For gigabyte tensors, serialization overhead becomes a bottleneck in distributed and multi-process workloads.

The standard approach copies tensor data multiple times: once to serialize, once to deserialize. At scale, this overhead adds up. Shared memory is the solution: allocate a block of memory in the kernel that multiple processes can access directly. Instead of sending the tensor data, you send just the metadata: shape, dtype, and the ID of the shared memory block. The receiving process attaches to that block and gets instant access. No copying.

## What This Project Does

A clean Python API over POSIX shared memory that handles tensor creation, attachment, lifecycle management, and cleanup. It wraps `multiprocessing.shared_memory` with a structured binary header format, automatic process-local tracking, and proper resource cleanup.

- **Zero-copy tensor sharing** between processes on the same machine via POSIX shared memory
- **Structured header format** with magic number validation, shape, dtype, creator PID, and timestamp
- **Automatic lifecycle management** with owner/non-owner semantics and garbage collection hooks
- **Process-local segment registry** for tracking and bulk cleanup of active shared memory blocks
- **Safety validation** for dtypes, shapes, and segment integrity on attachment

## Architecture

```mermaid
graph LR
    subgraph "Process A (Creator)"
        A1["NumPy Array"] --> A2["create_shared_tensor()"]
        A2 --> A3["Write Header + Data"]
    end

    subgraph "Kernel Shared Memory"
        SHM["POSIX /dev/shm<br/>Header | Shape | Dtype | Tensor Data"]
    end

    subgraph "Process B (Consumer)"
        B1["attach_shared_tensor()"] --> B2["Read Header"]
        B2 --> B3["NumPy View<br/>(zero-copy)"]
    end

    A3 --> SHM
    SHM --> B1

    style SHM fill:#fff9c4
```

The system uses a producer/consumer pattern. The producer calls `create_shared_tensor()`, which allocates a POSIX shared memory segment in `/dev/shm` (a tmpfs filesystem that lives entirely in RAM -- no disk I/O), writes a structured binary header (magic number, ndim, dtype, creator PID, timestamp), followed by the shape tuple and raw tensor data. The consumer calls `attach_shared_tensor()`, reads the header, validates the magic number to confirm the segment belongs to this library, and creates a NumPy array view directly into the shared buffer. Both processes now see the same physical RAM pages. The only cost is the TLB lookup to resolve virtual to physical addresses -- nanoseconds.

## Quick Start

```bash
git clone https://github.com/jrajath94/zero-copy-tensor-ipc.git
cd zero-copy-tensor-ipc
make install
```

```python
import numpy as np
from zero_copy_tensor_ipc import create_shared_tensor, attach_shared_tensor

# Process A: Create shared tensor
data = np.random.randn(10000, 10000).astype(np.float32)
tensor = create_shared_tensor("model_weights", data=data)
print(f"Created: {tensor}")
# SharedMemoryTensor(name='zctipc_model_weights', shape=(10000, 10000), dtype=float32, owner, open)

# Process B (separate Python process):
weights = attach_shared_tensor("model_weights")
print(f"Attached: shape={weights.shape}, dtype={weights.dtype}")
# Both processes now access the same memory region. No copying occurs.

# Cleanup
tensor.close()  # Owner unlinks the segment
```

## Key Results

This library enables true zero-copy tensor sharing between processes on the same machine via POSIX shared memory. Unlike standard multiprocessing queues that serialize and deserialize data, this approach achieves zero-copy access by mapping the same physical memory pages into multiple process address spaces.

For large tensors (gigabytes), shared memory eliminates the serialization bottleneck that occurs with standard multiprocessing queues. The cost is only the TLB lookup to resolve virtual-to-physical address translation.

## Design Decisions

| Decision                                                          | Rationale                                                                | Alternative Considered                               | Tradeoff                                                                 |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Structured binary header (magic + ndim + dtype + PID + timestamp) | Validates segment ownership on attach, prevents cross-library collisions | Send metadata through a side channel (queue)         | 56 bytes overhead per segment, but enables self-describing segments      |
| Owner/non-owner semantics                                         | Only the creator calls `shm_unlink`, preventing premature cleanup        | Reference counting across processes                  | Simpler, but consumer must not outlive producer without explicit handoff |
| Process-local segment registry                                    | Enables `cleanup_all_segments()` for emergency cleanup                   | Rely on OS cleanup on process exit                   | Small memory overhead, but prevents leaked segments on crashes           |
| Magic number validation (0x5A435449)                              | Rejects attachment to segments created by other libraries                | Trust the caller                                     | One extra struct unpack on attach -- negligible cost for safety          |
| NumPy-only integration                                            | Covers 95%+ of ML tensor use cases without additional dependencies       | Support PyTorch tensors, TensorFlow tensors directly | No GPU tensor support; requires CPU staging for GPU workflows            |

## How It Works

Under the hood, shared memory on Linux and macOS uses POSIX APIs: `shm_open`, `mmap`, and `shm_unlink`. The producer calls `shm_open` to create a named shared memory object in `/dev/shm` (on Linux) or the kernel's shared memory filesystem. This returns a file descriptor. The producer then calls `ftruncate` to set the size and `mmap` to map it into the process's virtual address space.

Python 3.8+ exposes this through `multiprocessing.shared_memory.SharedMemory`. This library adds a structured binary protocol on top:

```
Segment Layout:
[Header: 24 bytes]  magic(4B) | ndim(4B) | dtype_len(4B) | pid(4B) | timestamp(8B)
[Shape: ndim * 8B]  dim0(8B) | dim1(8B) | ...
[Dtype: 32B]        dtype string (e.g., "float32"), padded
[Data: N bytes]     raw tensor data, contiguous C-order
```

The `create_shared_tensor` function allocates the segment, writes the header and shape metadata using `struct.pack_into`, then creates a NumPy `ndarray` backed by the shared buffer starting at the data offset. If initial data is provided, a single `np.copyto` populates the shared memory. The `attach_shared_tensor` function opens the existing segment, reads and validates the header (checking the magic number), reconstructs the shape and dtype, and creates a new NumPy view into the same buffer.

The key insight: `/dev/shm` is a tmpfs filesystem that lives entirely in RAM. There is no disk I/O. The kernel manages page allocation and process-to-page mapping. When both processes mmap the same shared memory object, they share the same physical RAM pages. The CPU's MMU handles the virtual-to-physical translation in nanoseconds.

**Memory leak prevention** is critical. Shared memory blocks persist in the kernel until explicitly unlinked. If a process crashes before calling `shm_unlink`, the block stays allocated. The library addresses this with: `__del__` cleanup on garbage collection, a process-local `_ACTIVE_SEGMENTS` registry, and a `cleanup_all_segments()` function for emergency cleanup. In Docker containers, `/dev/shm` defaults to only 64MB -- you must set `--shm-size=16g` when running training containers, or allocation will fail.

## Testing

```bash
make test    # Unit + integration tests
pytest tests/ -v --cov=src/zero_copy_tensor_ipc
```

## Project Structure

```
zero-copy-tensor-ipc/
  src/zero_copy_tensor_ipc/
    ipc.py            # SharedMemoryTensor, create/attach functions, header protocol
    __init__.py       # Public API exports
  tests/
    test_core.py      # Shared memory creation, attachment, cleanup, edge cases
  docs/
    interview-prep.md # Development notes
```

## What I'd Improve

- **CUDA IPC integration.** NVIDIA's `cuda.IPCMemoryHandle` allows one process to allocate GPU memory and share the handle with another process on the same machine. Combined with CPU shared memory for metadata, this would give true end-to-end zero-copy for both CPU and GPU tensors -- no host memory staging.
- **Memory pool allocator.** Each tensor currently creates and destroys its own shared memory block. For high-frequency put/get patterns (thousands of batches per second), the overhead of `shm_open`/`shm_unlink` system calls adds up. A slab allocator that pre-allocates a large shared region and hands out fixed-size slots would amortize the system call cost.
- **PyTorch tensor wrapper.** `SharedMemoryTensor` currently works with NumPy only. Wrapping PyTorch tensors with `torch.from_numpy` on the shared buffer would enable direct integration with training loops without an intermediate copy.

## License

MIT -- Rajath John
