#!/usr/bin/env python3
import torch
import time

# Create 99% sparse tensor (2048, 16384)
sparse_shape = (2048, 2048 * 8)
total_elements = sparse_shape[0] * sparse_shape[1]
non_zero_elements = int(total_elements * 0.01)

indices = torch.randint(0, sparse_shape[0], (non_zero_elements,)), torch.randint(0, sparse_shape[1], (non_zero_elements,))
indices = torch.stack(indices)
values = torch.randn(non_zero_elements)
sparse_tensor = torch.sparse_coo_tensor(indices, values, sparse_shape).coalesce()

# Create dense tensor (16384, 2048)
dense_tensor = torch.randn(2048 * 8, 2048)

# Benchmark sparse @ dense
start = time.perf_counter()
for _ in range(10):
    result = torch.sparse.mm(sparse_tensor, dense_tensor)
end = time.perf_counter()

print(f"Sparse @ Dense: {(end - start) / 10:.6f}s")
print(f"Result shape: {result.shape}")