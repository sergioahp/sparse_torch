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

# Convert sparse to dense for comparison
sparse_dense = sparse_tensor.to_dense()

# Benchmark sparse @ dense
start = time.perf_counter()
for _ in range(10):
    result = torch.sparse.mm(sparse_tensor, dense_tensor)
end = time.perf_counter()
sparse_time = (end - start) / 10

# Benchmark dense @ dense
start = time.perf_counter()
for _ in range(10):
    result = torch.mm(sparse_dense, dense_tensor)
end = time.perf_counter()
dense_time = (end - start) / 10

print(f"Sparse @ Dense: {sparse_time:.6f}s")
print(f"Dense @ Dense:  {dense_time:.6f}s")
print(f"Speedup: {dense_time / sparse_time:.2f}x")
print(f"Result shape: {result.shape}")