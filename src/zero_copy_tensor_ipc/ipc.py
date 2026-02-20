"""Zero-copy tensor sharing via POSIX shared memory.

Enables efficient inter-process tensor communication without serialization
overhead. Tensors backed by shared memory can be accessed by multiple processes
simultaneously with zero-copy semantics.
"""

import logging
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Shared memory segment name prefix for namespacing
SHM_PREFIX = "zctipc_"

# Header layout: magic(4B) + ndim(4B) + dtype_len(4B) + pid(4B) + timestamp(8B)
HEADER_FORMAT = "<III I d"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Magic number to validate segments belong to this library
MAGIC_NUMBER = 0x5A435449  # "ZCTI"

# Maximum tensor dimensions supported
MAX_NDIM = 16

# Per-dimension shape entry: int64
SHAPE_ITEM_FORMAT = "<q"
SHAPE_ITEM_SIZE = struct.calcsize(SHAPE_ITEM_FORMAT)

# Maximum dtype string length stored after shape
DTYPE_STR_MAX_LEN = 32

# Process-local registry of active shared memory segments
_ACTIVE_SEGMENTS: Dict[str, "SharedMemoryTensor"] = {}


@dataclass(frozen=True)
class TensorMetadata:
    """Immutable metadata describing a shared tensor.

    Attributes:
        name: Unique identifier for the shared memory segment.
        shape: Tuple of dimension sizes.
        dtype: NumPy dtype string (e.g., 'float32').
        creator_pid: PID of the process that created the segment.
        created_at: Unix timestamp of creation.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: str
    creator_pid: int
    created_at: float


class SharedMemoryTensor:
    """A numpy array backed by POSIX shared memory for zero-copy IPC.

    Manages the lifecycle of shared memory segments containing tensor data.
    The creating process owns the segment; attaching processes get read/write
    access without copying data.

    Attributes:
        metadata: Descriptor of the shared tensor.
        array: NumPy ndarray view into shared memory.
    """

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        array: np.ndarray,
        metadata: TensorMetadata,
        is_owner: bool,
    ) -> None:
        """Initialize a shared memory tensor wrapper.

        Args:
            shm: The underlying shared memory object.
            array: NumPy array view into the shared memory buffer.
            metadata: Tensor metadata (shape, dtype, etc.).
            is_owner: Whether this instance owns (created) the segment.
        """
        self._shm = shm
        self._array = array
        self._metadata = metadata
        self._is_owner = is_owner
        self._closed = False

    @property
    def metadata(self) -> TensorMetadata:
        """Return tensor metadata."""
        return self._metadata

    @property
    def array(self) -> np.ndarray:
        """Return the numpy array view into shared memory."""
        return self._array

    @property
    def name(self) -> str:
        """Return the shared memory segment name."""
        return self._metadata.name

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the tensor shape."""
        return self._metadata.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the tensor dtype."""
        return np.dtype(self._metadata.dtype)

    @property
    def is_owner(self) -> bool:
        """Return whether this instance owns the shared memory."""
        return self._is_owner

    @property
    def nbytes(self) -> int:
        """Return the size of the tensor data in bytes."""
        return self._array.nbytes

    def close(self) -> None:
        """Close access to the shared memory segment.

        Detaches from shared memory. If this instance is the owner,
        also unlinks (destroys) the segment.
        """
        if self._closed:
            return

        self._closed = True
        segment_name = self._metadata.name

        try:
            self._shm.close()
            logger.debug("Closed shared memory segment: %s", segment_name)
        except Exception:
            logger.exception("Error closing segment: %s", segment_name)

        if self._is_owner:
            self._unlink_segment(segment_name)

        _ACTIVE_SEGMENTS.pop(segment_name, None)

    def _unlink_segment(self, segment_name: str) -> None:
        """Unlink (destroy) the shared memory segment.

        Args:
            segment_name: Name of the segment to unlink.
        """
        try:
            self._shm.unlink()
            logger.info("Unlinked shared memory segment: %s", segment_name)
        except FileNotFoundError:
            logger.debug("Segment already unlinked: %s", segment_name)
        except Exception:
            logger.exception("Error unlinking segment: %s", segment_name)

    def __del__(self) -> None:
        """Ensure shared memory is cleaned up on garbage collection."""
        self.close()

    def __repr__(self) -> str:
        """Return human-readable representation."""
        status = "closed" if self._closed else "open"
        owner_str = "owner" if self._is_owner else "attached"
        return (
            f"SharedMemoryTensor(name='{self.name}', shape={self.shape}, "
            f"dtype={self.dtype}, {owner_str}, {status})"
        )


def _compute_segment_size(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    """Compute total shared memory size including header + shape + dtype + data.

    Args:
        shape: Tensor dimensions.
        dtype: NumPy dtype.

    Returns:
        Total bytes needed for the shared memory segment.
    """
    data_size = int(np.prod(shape)) * dtype.itemsize
    shape_size = len(shape) * SHAPE_ITEM_SIZE
    return HEADER_SIZE + shape_size + DTYPE_STR_MAX_LEN + data_size


def _write_header(
    buf: memoryview,
    ndim: int,
    dtype_str: str,
    shape: Tuple[int, ...],
    pid: int,
    timestamp: float,
) -> int:
    """Write tensor metadata header into shared memory buffer.

    Args:
        buf: Writable memory view into shared memory.
        ndim: Number of dimensions.
        dtype_str: Dtype as string (e.g. 'float32').
        shape: Tensor shape tuple.
        pid: Creator process ID.
        timestamp: Creation time.

    Returns:
        Byte offset where tensor data begins.
    """
    dtype_len = len(dtype_str)
    struct.pack_into(HEADER_FORMAT, buf, 0, MAGIC_NUMBER, ndim, dtype_len, pid, timestamp)

    offset = HEADER_SIZE
    for dim in shape:
        struct.pack_into(SHAPE_ITEM_FORMAT, buf, offset, dim)
        offset += SHAPE_ITEM_SIZE

    # Write dtype string after shape
    dtype_bytes = dtype_str.encode("ascii")
    buf[offset : offset + len(dtype_bytes)] = dtype_bytes
    offset += DTYPE_STR_MAX_LEN

    return offset


def _read_header(
    buf: memoryview,
) -> Tuple[Tuple[int, ...], str, int, float]:
    """Read tensor metadata header from shared memory buffer.

    Args:
        buf: Readable memory view into shared memory.

    Returns:
        Tuple of (shape, dtype_str, creator_pid, timestamp).

    Raises:
        ValueError: If magic number doesn't match.
    """
    magic, ndim, dtype_len, pid, timestamp = struct.unpack_from(
        HEADER_FORMAT, buf, 0
    )
    _validate_magic(magic)

    offset = HEADER_SIZE
    shape = _read_shape(buf, offset, ndim)
    offset += ndim * SHAPE_ITEM_SIZE

    # Read dtype string
    dtype_bytes = bytes(buf[offset : offset + dtype_len])
    dtype_str = dtype_bytes.decode("ascii")

    return shape, dtype_str, pid, timestamp


def _validate_magic(magic: int) -> None:
    """Validate the magic number from a shared memory header.

    Args:
        magic: The magic number read from the buffer.

    Raises:
        ValueError: If magic number doesn't match expected value.
    """
    if magic != MAGIC_NUMBER:
        raise ValueError(
            f"Invalid magic number 0x{magic:08X}, expected 0x{MAGIC_NUMBER:08X}. "
            "Segment was not created by zero-copy-tensor-ipc."
        )


def _read_shape(
    buf: memoryview, offset: int, ndim: int
) -> Tuple[int, ...]:
    """Read shape tuple from buffer at given offset.

    Args:
        buf: Memory view to read from.
        offset: Starting byte offset.
        ndim: Number of dimensions to read.

    Returns:
        Shape as tuple of ints.
    """
    shape: List[int] = []
    for _ in range(ndim):
        (dim,) = struct.unpack_from(SHAPE_ITEM_FORMAT, buf, offset)
        shape.append(dim)
        offset += SHAPE_ITEM_SIZE
    return tuple(shape)


def _validate_shape(shape: Tuple[int, ...]) -> None:
    """Validate tensor shape.

    Args:
        shape: Tuple of dimension sizes.

    Raises:
        ValueError: If shape is invalid (empty, too many dims, or negative).
    """
    if not shape:
        raise ValueError("Shape must be non-empty")
    if len(shape) > MAX_NDIM:
        raise ValueError(f"Too many dimensions: {len(shape)} > {MAX_NDIM}")
    for i, dim in enumerate(shape):
        if dim <= 0:
            raise ValueError(f"Dimension {i} must be positive, got {dim}")


def _validate_dtype(dtype: np.dtype) -> None:
    """Validate numpy dtype for shared memory compatibility.

    Args:
        dtype: NumPy dtype to validate.

    Raises:
        TypeError: If dtype is not a simple numeric type.
    """
    allowed_kinds = {"f", "i", "u", "b"}  # float, int, uint, bool
    if dtype.kind not in allowed_kinds:
        raise TypeError(
            f"Unsupported dtype kind '{dtype.kind}' ({dtype}). "
            f"Must be one of: float, int, uint, bool."
        )


def create_shared_tensor(
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: str = "float32",
    data: Optional[np.ndarray] = None,
) -> SharedMemoryTensor:
    """Create a new shared memory tensor.

    Allocates a POSIX shared memory segment and maps a numpy array onto it.
    The calling process becomes the owner responsible for cleanup.

    Args:
        name: Unique name for the shared memory segment.
        shape: Desired tensor shape. Inferred from data if not provided.
        dtype: NumPy dtype string (default: 'float32').
        data: Optional initial data to copy into shared memory.

    Returns:
        SharedMemoryTensor with ownership of the underlying segment.

    Raises:
        ValueError: If shape is invalid, or both shape and data are missing.
        TypeError: If dtype is unsupported.
    """
    shape, np_dtype = _resolve_shape_and_dtype(shape, dtype, data)
    _validate_shape(shape)
    _validate_dtype(np_dtype)

    segment_name = f"{SHM_PREFIX}{name}"
    total_size = _compute_segment_size(shape, np_dtype)

    logger.info(
        "Creating shared tensor '%s': shape=%s, dtype=%s, size=%d bytes",
        name, shape, dtype, total_size,
    )

    shm = shared_memory.SharedMemory(
        name=segment_name, create=True, size=total_size
    )

    try:
        return _initialize_tensor(shm, shape, np_dtype, data, segment_name)
    except Exception:
        shm.close()
        shm.unlink()
        raise


def _resolve_shape_and_dtype(
    shape: Optional[Tuple[int, ...]],
    dtype: str,
    data: Optional[np.ndarray],
) -> Tuple[Tuple[int, ...], np.dtype]:
    """Resolve final shape and dtype from arguments.

    Args:
        shape: Explicit shape or None.
        dtype: Dtype string.
        data: Optional data array.

    Returns:
        Tuple of (resolved_shape, resolved_dtype).

    Raises:
        ValueError: If neither shape nor data is provided, or mismatch.
    """
    if data is not None:
        resolved_shape = shape if shape is not None else data.shape
        if shape is not None and data.shape != shape:
            raise ValueError(
                f"Data shape {data.shape} doesn't match requested shape {shape}"
            )
        return resolved_shape, np.dtype(dtype)

    if shape is None:
        raise ValueError("Either shape or data must be provided")

    return shape, np.dtype(dtype)


def _initialize_tensor(
    shm: shared_memory.SharedMemory,
    shape: Tuple[int, ...],
    np_dtype: np.dtype,
    data: Optional[np.ndarray],
    segment_name: str,
) -> SharedMemoryTensor:
    """Initialize shared memory with header and data.

    Args:
        shm: Allocated shared memory object.
        shape: Tensor shape.
        np_dtype: NumPy dtype.
        data: Optional data to copy in.
        segment_name: Full segment name.

    Returns:
        Fully initialized SharedMemoryTensor.
    """
    buf = shm.buf
    pid = os.getpid()
    timestamp = time.time()
    dtype_str = str(np_dtype)

    data_offset = _write_header(
        buf, len(shape), dtype_str, shape, pid, timestamp
    )

    array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf[data_offset:])

    if data is not None:
        np.copyto(array, data.astype(np_dtype))
    else:
        array[:] = 0

    metadata = TensorMetadata(
        name=segment_name,
        shape=shape,
        dtype=dtype_str,
        creator_pid=pid,
        created_at=timestamp,
    )

    tensor = SharedMemoryTensor(
        shm=shm, array=array, metadata=metadata, is_owner=True
    )
    _ACTIVE_SEGMENTS[segment_name] = tensor
    logger.info("Created shared tensor '%s' (pid=%d)", segment_name, pid)
    return tensor


def attach_shared_tensor(name: str) -> SharedMemoryTensor:
    """Attach to an existing shared memory tensor.

    Opens an existing shared memory segment and maps a numpy array view
    onto it. The caller does NOT own the segment and will not unlink it.

    Args:
        name: Name used when creating the tensor (without prefix).

    Returns:
        SharedMemoryTensor attached to the existing segment (non-owner).

    Raises:
        FileNotFoundError: If no segment with this name exists.
        ValueError: If the segment header is invalid/corrupt.
    """
    segment_name = f"{SHM_PREFIX}{name}"
    logger.info("Attaching to shared tensor '%s'", segment_name)

    shm = shared_memory.SharedMemory(name=segment_name, create=False)

    try:
        return _build_attached_tensor(shm, segment_name)
    except Exception:
        shm.close()
        raise


def _build_attached_tensor(
    shm: shared_memory.SharedMemory,
    segment_name: str,
) -> SharedMemoryTensor:
    """Build a SharedMemoryTensor from an attached segment.

    Args:
        shm: Already-opened shared memory object.
        segment_name: Full segment name.

    Returns:
        SharedMemoryTensor with is_owner=False.
    """
    shape, dtype_str, creator_pid, timestamp = _read_header(shm.buf)
    np_dtype = np.dtype(dtype_str)

    data_offset = (
        HEADER_SIZE + len(shape) * SHAPE_ITEM_SIZE + DTYPE_STR_MAX_LEN
    )
    array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf[data_offset:])

    metadata = TensorMetadata(
        name=segment_name,
        shape=shape,
        dtype=dtype_str,
        creator_pid=creator_pid,
        created_at=timestamp,
    )

    tensor = SharedMemoryTensor(
        shm=shm, array=array, metadata=metadata, is_owner=False
    )
    _ACTIVE_SEGMENTS[segment_name] = tensor
    logger.info(
        "Attached to shared tensor '%s': shape=%s, dtype=%s",
        segment_name, shape, np_dtype,
    )
    return tensor


def list_active_segments() -> List[TensorMetadata]:
    """List all active shared memory tensor segments in this process.

    Returns:
        List of TensorMetadata for currently active segments.
    """
    return [tensor.metadata for tensor in _ACTIVE_SEGMENTS.values()]


def cleanup_all_segments() -> int:
    """Close and unlink all active shared memory segments.

    Returns:
        Number of segments cleaned up.
    """
    segment_names = list(_ACTIVE_SEGMENTS.keys())
    count = 0
    for name in segment_names:
        tensor = _ACTIVE_SEGMENTS.get(name)
        if tensor is not None:
            tensor.close()
            count += 1
            logger.info("Cleaned up segment: %s", name)
    return count
