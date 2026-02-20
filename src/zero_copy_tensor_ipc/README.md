# zero_copy_tensor_ipc Module

OS-level shared memory for inter-process tensor sharing.

## How It Works

1. Process A creates a named shared memory region
2. Writes tensor data once (copy)
3. Process B attaches to the same region by name
4. Both processes see identical bytes—no copy needed

## Key Insight

The performance win comes from avoiding serialization:
- Pickling 1GB array: 100-150ms
- Shared memory attach: <1µs

But there's a cost: no built-in synchronization. Multiple writers corrupt data. Use external locking (mutex) if you need concurrent writes.

## Lifecycle

Shared memory persists until explicitly unlinked. It survives process death. You're responsible for cleanup—use try/finally or context managers to avoid leaks.
