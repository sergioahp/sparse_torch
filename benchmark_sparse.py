#!/usr/bin/env python3
import torch
import torch.sparse
import time
import os
from typing import Tuple, Dict, Any

try:
    import torch_sparse
    HAS_TORCH_SPARSE = True
    print("torch-sparse available")
except ImportError:
    HAS_TORCH_SPARSE = False
    print("torch-sparse not available, using built-in sparse ops only")

def create_sparse_tensor(shape: Tuple[int, int], sparsity: float = 0.99) -> torch.Tensor:
    """Create a sparse tensor with specified sparsity (percentage of zeros)"""
    total_elements = shape[0] * shape[1]
    non_zero_elements = int(total_elements * (1 - sparsity))
    
    # Generate random indices for non-zero elements
    indices = torch.randint(0, shape[0], (non_zero_elements,)), torch.randint(0, shape[1], (non_zero_elements,))
    indices = torch.stack(indices)
    
    # Generate random values for non-zero elements
    values = torch.randn(non_zero_elements)
    
    # Create sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor.coalesce()

def create_torch_sparse_tensor(shape: Tuple[int, int], sparsity: float = 0.99):
    """Create a torch-sparse SparseTensor with specified sparsity"""
    if not HAS_TORCH_SPARSE:
        return None
        
    total_elements = shape[0] * shape[1]
    non_zero_elements = int(total_elements * (1 - sparsity))
    
    # Generate random indices for non-zero elements
    row = torch.randint(0, shape[0], (non_zero_elements,))
    col = torch.randint(0, shape[1], (non_zero_elements,))
    
    # Generate random values for non-zero elements
    value = torch.randn(non_zero_elements)
    
    # Create torch-sparse SparseTensor
    return torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=shape)

def create_dense_tensor(shape: Tuple[int, int]) -> torch.Tensor:
    """Create a dense tensor with random values"""
    return torch.randn(shape)

def benchmark_operation(op_name: str, operation, *args, warmup: int = 5, runs: int = 10) -> Dict[str, Any]:
    """Benchmark a tensor operation"""
    print(f"Benchmarking {op_name}...")
    
    # Warmup
    for _ in range(warmup):
        result = operation(*args)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(runs):
        result = operation(*args)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / runs
    
    return {
        'avg_time': avg_time,
        'result_shape': result.shape if hasattr(result, 'shape') else None,
        'result_is_sparse': result.is_sparse if hasattr(result, 'is_sparse') else False
    }

def main():
    """Main benchmarking function"""
    print("PyTorch Sparse Tensor Benchmarking")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tensors as specified
    print("Creating tensors...")
    sparse_shape = (2048, 2048 * 8)  # (2048, 16384)
    dense_shape = (2048 * 8, 2048)   # (16384, 2048)
    
    # 99% sparse tensor
    sparse_tensor = create_sparse_tensor(sparse_shape, sparsity=0.99).to(device)
    print(f"Sparse tensor shape: {sparse_tensor.shape}")
    print(f"Sparse tensor nnz: {sparse_tensor._nnz()}")
    print(f"Sparse tensor density: {sparse_tensor._nnz() / (sparse_tensor.shape[0] * sparse_tensor.shape[1]):.4f}")
    
    # Dense tensor
    dense_tensor = create_dense_tensor(dense_shape).to(device)
    print(f"Dense tensor shape: {dense_tensor.shape}")
    
    # Convert sparse to dense for comparison
    sparse_dense = sparse_tensor.to_dense()
    
    print("\nBenchmarking Operations:")
    print("-" * 30)
    
    # Matrix multiplication: sparse @ dense
    print("Matrix multiplication (sparse @ dense):")
    sparse_mm_result = benchmark_operation(
        "sparse @ dense",
        torch.sparse.mm,
        sparse_tensor,
        dense_tensor
    )
    print(f"  Time: {sparse_mm_result['avg_time']:.6f}s")
    
    # Matrix multiplication: dense @ dense (for comparison)
    print("\nMatrix multiplication (dense @ dense):")
    dense_mm_result = benchmark_operation(
        "dense @ dense",
        torch.mm,
        sparse_dense,
        dense_tensor
    )
    print(f"  Time: {dense_mm_result['avg_time']:.6f}s")
    
    # Element-wise operations
    print("\nElement-wise addition (sparse + sparse):")
    sparse_tensor_2 = create_sparse_tensor(sparse_shape, sparsity=0.99).to(device)
    sparse_add_result = benchmark_operation(
        "sparse + sparse",
        torch.add,
        sparse_tensor,
        sparse_tensor_2
    )
    print(f"  Time: {sparse_add_result['avg_time']:.6f}s")
    
    # torch-sparse operations if available
    if HAS_TORCH_SPARSE:
        print("\nTorch-sparse operations:")
        print("-" * 25)
        
        # Create torch-sparse tensor
        torch_sparse_tensor = create_torch_sparse_tensor(sparse_shape, sparsity=0.99)
        if device.type == 'cuda':
            torch_sparse_tensor = torch_sparse_tensor.cuda()
        
        print(f"Torch-sparse tensor shape: {torch_sparse_tensor.sparse_sizes()}")
        print(f"Torch-sparse tensor nnz: {torch_sparse_tensor.nnz()}")
        
        # torch-sparse matrix multiplication
        print("\nMatrix multiplication (torch-sparse @ dense):")
        def torch_sparse_matmul():
            return torch_sparse_tensor @ dense_tensor
        
        torch_sparse_mm_result = benchmark_operation(
            "torch-sparse @ dense",
            torch_sparse_matmul
        )
        print(f"  Time: {torch_sparse_mm_result['avg_time']:.6f}s")
        
        # Compare with built-in sparse
        speedup_vs_builtin = sparse_mm_result['avg_time'] / torch_sparse_mm_result['avg_time']
        print(f"  Speedup vs built-in sparse: {speedup_vs_builtin:.2f}x")
    
    # Memory usage comparison
    print("\nMemory Usage Comparison:")
    print("-" * 25)
    sparse_memory = sparse_tensor.element_size() * sparse_tensor._nnz() * 2  # indices + values
    dense_memory = sparse_dense.element_size() * sparse_dense.numel()
    
    print(f"Sparse tensor memory: {sparse_memory / 1024 / 1024:.2f} MB")
    print(f"Dense tensor memory: {dense_memory / 1024 / 1024:.2f} MB")
    print(f"Memory savings: {(1 - sparse_memory / dense_memory) * 100:.1f}%")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    print("-" * 23)
    speedup = dense_mm_result['avg_time'] / sparse_mm_result['avg_time']
    print(f"Matrix multiplication speedup: {speedup:.2f}x")
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()