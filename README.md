# zero-copy-tensor-ipc

Zero-copy tensor sharing via shared memory for inter-process communication.

## Quick Start
```bash
make install && make test
```

## Usage
```python
import numpy as np
from zero_copy_tensor_ipc import create_shared_tensor, attach_shared_tensor

# Create
data = np.random.randn(1000, 1000)
tensor = create_shared_tensor("my_tensor", data)

# Attach from another process
arr = attach_shared_tensor("my_tensor", (1000, 1000), "float64")
```
