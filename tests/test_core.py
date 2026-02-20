"""Tests for zero-copy tensor IPC shared memory operations."""

import os
import uuid

import numpy as np
import pytest

from zero_copy_tensor_ipc.ipc import (
    SharedMemoryTensor,
    TensorMetadata,
    create_shared_tensor,
    attach_shared_tensor,
    list_active_segments,
    cleanup_all_segments,
    _validate_shape,
    _validate_dtype,
    _compute_segment_size,
    _ACTIVE_SEGMENTS,
)


def _unique_name() -> str:
    """Generate a unique segment name for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def cleanup_segments():
    """Ensure all shared memory segments are cleaned up after each test."""
    yield
    cleanup_all_segments()


class TestCreateSharedTensor:
    """Tests for create_shared_tensor function."""

    def test_create_basic_tensor(self) -> None:
        """Create a basic float32 tensor and verify shape and dtype."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(4, 4), dtype="float32")

        assert tensor.shape == (4, 4)
        assert tensor.dtype == np.dtype("float32")
        assert tensor.is_owner is True
        assert tensor.array.shape == (4, 4)
        np.testing.assert_array_equal(tensor.array, np.zeros((4, 4)))

    def test_create_with_initial_data(self) -> None:
        """Create tensor pre-populated with data."""
        name = _unique_name()
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        tensor = create_shared_tensor(name, data=data, dtype="float64")

        np.testing.assert_array_almost_equal(tensor.array, data)
        assert tensor.shape == (3, 4)

    def test_create_infers_shape_from_data(self) -> None:
        """Shape is inferred from data when not explicitly provided."""
        name = _unique_name()
        data = np.ones((5, 3), dtype=np.float32)
        tensor = create_shared_tensor(name, data=data)

        assert tensor.shape == (5, 3)

    @pytest.mark.parametrize(
        "dtype",
        ["float32", "float64", "int32", "int64", "uint8"],
    )
    def test_create_various_dtypes(self, dtype: str) -> None:
        """Tensor creation works across numeric dtypes."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(10,), dtype=dtype)

        assert tensor.dtype == np.dtype(dtype)

    def test_create_1d_tensor(self) -> None:
        """Create a 1-dimensional tensor."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(100,))

        assert tensor.shape == (100,)
        assert tensor.nbytes == 100 * 4  # float32

    def test_create_3d_tensor(self) -> None:
        """Create a 3-dimensional tensor."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(2, 3, 4))

        assert tensor.shape == (2, 3, 4)

    def test_shape_data_mismatch_raises(self) -> None:
        """Mismatched shape and data shape raises ValueError."""
        name = _unique_name()
        data = np.zeros((3, 3))

        with pytest.raises(ValueError, match="doesn't match"):
            create_shared_tensor(name, shape=(4, 4), data=data)

    def test_no_shape_no_data_raises(self) -> None:
        """Missing both shape and data raises ValueError."""
        name = _unique_name()

        with pytest.raises(ValueError, match="Either shape or data"):
            create_shared_tensor(name)


class TestAttachSharedTensor:
    """Tests for attach_shared_tensor function."""

    def test_attach_reads_correct_data(self) -> None:
        """Attached tensor reads the same data written by creator."""
        name = _unique_name()
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        owner = create_shared_tensor(name, data=data)

        attached = attach_shared_tensor(name)

        np.testing.assert_array_equal(attached.array, data)
        assert attached.is_owner is False
        assert attached.shape == (4,)

        # Clean up attached before owner
        attached.close()

    def test_attach_sees_mutations(self) -> None:
        """Changes made via the owner are visible to attached tensors."""
        name = _unique_name()
        owner = create_shared_tensor(name, shape=(4,), dtype="float32")
        attached = attach_shared_tensor(name)

        owner.array[0] = 42.0
        assert attached.array[0] == 42.0

        attached.close()

    def test_attach_nonexistent_raises(self) -> None:
        """Attaching to a non-existent segment raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            attach_shared_tensor("nonexistent_segment_xyz")


class TestSharedMemoryTensorLifecycle:
    """Tests for SharedMemoryTensor close/cleanup behavior."""

    def test_close_owner_unlinks(self) -> None:
        """Closing the owner unlinks the shared memory segment."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(4,))
        tensor.close()

        # After unlink, attaching should fail
        with pytest.raises(FileNotFoundError):
            attach_shared_tensor(name)

    def test_close_is_idempotent(self) -> None:
        """Calling close multiple times doesn't raise."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(4,))
        tensor.close()
        tensor.close()  # Should not raise

    def test_repr_shows_state(self) -> None:
        """Repr includes name, shape, dtype, and status."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(2, 3))
        repr_str = repr(tensor)

        assert "shape=(2, 3)" in repr_str
        assert "owner" in repr_str
        assert "open" in repr_str

    def test_metadata_properties(self) -> None:
        """Metadata is correctly populated on creation."""
        name = _unique_name()
        tensor = create_shared_tensor(name, shape=(8,), dtype="float64")
        meta = tensor.metadata

        assert meta.shape == (8,)
        assert meta.dtype == "float64"
        assert meta.creator_pid == os.getpid()
        assert meta.created_at > 0


class TestSegmentRegistry:
    """Tests for the active segment registry."""

    def test_list_active_segments(self) -> None:
        """Active segments are tracked in the registry."""
        name1 = _unique_name()
        name2 = _unique_name()
        t1 = create_shared_tensor(name1, shape=(4,))
        t2 = create_shared_tensor(name2, shape=(8,))

        active = list_active_segments()
        names = [m.name for m in active]

        assert f"zctipc_{name1}" in names
        assert f"zctipc_{name2}" in names

    def test_cleanup_all_segments(self) -> None:
        """cleanup_all_segments closes and removes all tracked segments."""
        name1 = _unique_name()
        name2 = _unique_name()
        create_shared_tensor(name1, shape=(4,))
        create_shared_tensor(name2, shape=(4,))

        count = cleanup_all_segments()
        assert count == 2
        assert len(list_active_segments()) == 0


class TestValidation:
    """Tests for input validation helpers."""

    def test_empty_shape_raises(self) -> None:
        """Empty shape tuple is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_shape(())

    def test_negative_dimension_raises(self) -> None:
        """Negative dimension is rejected."""
        with pytest.raises(ValueError, match="positive"):
            _validate_shape((4, -1))

    def test_zero_dimension_raises(self) -> None:
        """Zero dimension is rejected."""
        with pytest.raises(ValueError, match="positive"):
            _validate_shape((0, 4))

    @pytest.mark.parametrize(
        "dtype_str",
        ["float32", "float64", "int32", "int64", "uint8", "bool"],
    )
    def test_valid_dtypes_accepted(self, dtype_str: str) -> None:
        """Numeric dtypes are accepted."""
        _validate_dtype(np.dtype(dtype_str))  # Should not raise

    def test_string_dtype_rejected(self) -> None:
        """String dtype is rejected."""
        with pytest.raises(TypeError, match="Unsupported dtype"):
            _validate_dtype(np.dtype("U10"))

    def test_segment_size_computation(self) -> None:
        """Segment size includes header + shape + dtype + data."""
        size = _compute_segment_size((100,), np.dtype("float32"))
        # Data alone is 400 bytes, total should be larger due to header
        assert size > 400
