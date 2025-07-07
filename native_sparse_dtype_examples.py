#!/usr/bin/env python3
"""
Examples of using PyTorch native sparse tensors with different dtypes.
This complements the torch-sparse examples and shows native PyTorch sparse functionality.
"""

import torch
import torch.nn as nn
import torch.sparse
import time
from typing import Tuple, Dict, Any

def create_native_sparse_tensor(shape: Tuple[int, int], sparsity: float = 0.99, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a native PyTorch sparse COO tensor with specified dtype and sparsity"""
    total_elements = shape[0] * shape[1]
    non_zero_elements = int(total_elements * (1 - sparsity))
    
    # Generate random indices for non-zero elements
    indices = torch.randint(0, shape[0], (non_zero_elements,)), torch.randint(0, shape[1], (non_zero_elements,))
    indices = torch.stack(indices)
    
    # Generate random values with specified dtype
    values = torch.randn(non_zero_elements, dtype=dtype)
    
    # Create sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)
    return sparse_tensor.coalesce()

def demonstrate_native_sparse_creation():
    """Demonstrate creating native sparse tensors with different dtypes"""
    print("=== Native PyTorch Sparse Tensor Creation ===")
    
    shape = (1000, 1000)
    sparsity = 0.95
    
    # Create sparse tensors with different dtypes
    sparse_fp32 = create_native_sparse_tensor(shape, sparsity, torch.float32)
    sparse_fp16 = create_native_sparse_tensor(shape, sparsity, torch.float16)
    sparse_fp64 = create_native_sparse_tensor(shape, sparsity, torch.float64)
    
    print(f"FP32 sparse tensor: shape={sparse_fp32.shape}, dtype={sparse_fp32.dtype}, nnz={sparse_fp32._nnz()}")
    print(f"FP16 sparse tensor: shape={sparse_fp16.shape}, dtype={sparse_fp16.dtype}, nnz={sparse_fp16._nnz()}")
    print(f"FP64 sparse tensor: shape={sparse_fp64.shape}, dtype={sparse_fp64.dtype}, nnz={sparse_fp64._nnz()}")
    
    # BF16 if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        sparse_bf16 = create_native_sparse_tensor(shape, sparsity, torch.bfloat16)
        print(f"BF16 sparse tensor: shape={sparse_bf16.shape}, dtype={sparse_bf16.dtype}, nnz={sparse_bf16._nnz()}")
    else:
        print("BF16 not supported on this device")

def demonstrate_native_sparse_conversion():
    """Demonstrate dtype conversion with native sparse tensors"""
    print("\\n=== Native Sparse Tensor dtype Conversion ===")
    
    # Create a base sparse tensor
    sparse_fp32 = create_native_sparse_tensor((500, 500), 0.9, torch.float32)
    print(f"Original: dtype={sparse_fp32.dtype}, nnz={sparse_fp32._nnz()}")
    
    # Convert to different dtypes
    sparse_fp16 = sparse_fp32.half()
    sparse_fp64 = sparse_fp32.double()
    
    print(f"After .half(): dtype={sparse_fp16.dtype}, nnz={sparse_fp16._nnz()}")
    print(f"After .double(): dtype={sparse_fp64.dtype}, nnz={sparse_fp64._nnz()}")
    
    # Convert back to FP32
    sparse_fp32_back = sparse_fp16.float()
    print(f"Back to float: dtype={sparse_fp32_back.dtype}, nnz={sparse_fp32_back._nnz()}")
    
    # Using .to() method for conversion
    sparse_to_fp16 = sparse_fp32.to(torch.float16)
    print(f"Using .to(torch.float16): dtype={sparse_to_fp16.dtype}, nnz={sparse_to_fp16._nnz()}")

def demonstrate_native_sparse_matmul():
    """Demonstrate matrix multiplication with native sparse tensors"""
    print("\\n=== Native Sparse Matrix Multiplication ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sparse and dense tensors
    sparse_shape = (1024, 2048)
    dense_shape = (2048, 512)
    sparsity = 0.95
    
    # Different precision combinations
    sparse_fp32 = create_native_sparse_tensor(sparse_shape, sparsity, torch.float32).to(device)
    sparse_fp16 = create_native_sparse_tensor(sparse_shape, sparsity, torch.float16).to(device)
    
    dense_fp32 = torch.randn(dense_shape, dtype=torch.float32, device=device)
    dense_fp16 = torch.randn(dense_shape, dtype=torch.float16, device=device)
    
    # FP32 @ FP32
    print("\\nFP32 sparse @ FP32 dense:")
    start = time.perf_counter()
    result_fp32 = torch.sparse.mm(sparse_fp32, dense_fp32)
    end = time.perf_counter()
    print(f"  Result dtype: {result_fp32.dtype}, shape: {result_fp32.shape}")
    print(f"  Time: {end - start:.4f}s")
    
    # FP16 @ FP16
    print("\\nFP16 sparse @ FP16 dense:")
    start = time.perf_counter()
    result_fp16 = torch.sparse.mm(sparse_fp16, dense_fp16)
    end = time.perf_counter()
    print(f"  Result dtype: {result_fp16.dtype}, shape: {result_fp16.shape}")
    print(f"  Time: {end - start:.4f}s")
    
    # Mixed precision: FP16 @ FP32 (requires conversion)
    print("\\nMixed precision FP16 sparse @ FP32 dense (with conversion):")
    start = time.perf_counter()
    # Convert sparse tensor to match dense tensor dtype
    sparse_fp16_to_fp32 = sparse_fp16.float()
    result_mixed = torch.sparse.mm(sparse_fp16_to_fp32, dense_fp32)
    end = time.perf_counter()
    print(f"  Result dtype: {result_mixed.dtype}, shape: {result_mixed.shape}")
    print(f"  Time: {end - start:.4f}s")
    print("  Note: Native PyTorch sparse ops require matching dtypes")

def demonstrate_autocast_native_sparse():
    """Demonstrate autocast with native sparse tensors"""
    print("\\n=== Autocast with Native Sparse Tensors ===")
    
    if not torch.cuda.is_available():
        print("Skipping - CUDA not available for autocast demo")
        return
    
    device = torch.device('cuda')
    
    # Create tensors
    sparse_fp32 = create_native_sparse_tensor((1024, 2048), 0.95, torch.float32).to(device)
    dense_fp32 = torch.randn(2048, 512, dtype=torch.float32, device=device)
    
    print("Without autocast:")
    result_no_autocast = torch.sparse.mm(sparse_fp32, dense_fp32)
    print(f"  Result dtype: {result_no_autocast.dtype}")
    
    print("\\nWith autocast:")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result_autocast = torch.sparse.mm(sparse_fp32, dense_fp32)
        print(f"  Result dtype: {result_autocast.dtype}")

def demonstrate_sparse_csr_dtypes():
    """Demonstrate CSR sparse tensor dtypes"""
    print("\\n=== CSR Sparse Tensor dtypes ===")
    
    # Create a simple CSR tensor
    crow_indices = torch.tensor([0, 2, 4])
    col_indices = torch.tensor([0, 1, 0, 1])
    values_fp32 = torch.tensor([1., 2., 3., 4.], dtype=torch.float32)
    values_fp16 = torch.tensor([1., 2., 3., 4.], dtype=torch.float16)
    
    # Create CSR tensors with different dtypes
    csr_fp32 = torch.sparse_csr_tensor(crow_indices, col_indices, values_fp32, dtype=torch.float32)
    csr_fp16 = torch.sparse_csr_tensor(crow_indices, col_indices, values_fp16, dtype=torch.float16)
    
    print(f"CSR FP32: shape={csr_fp32.shape}, dtype={csr_fp32.dtype}, nnz={csr_fp32._nnz()}")
    print(f"CSR FP16: shape={csr_fp16.shape}, dtype={csr_fp16.dtype}, nnz={csr_fp16._nnz()}")
    
    # Convert between dtypes
    csr_fp32_to_fp16 = csr_fp32.half()
    print(f"CSR FP32 -> FP16: dtype={csr_fp32_to_fp16.dtype}")

def demonstrate_gradient_computation():
    """Demonstrate gradient computation with different dtypes"""
    print("\\n=== Gradient Computation with Sparse Tensors ===")
    
    # Create a sparse tensor that requires gradients
    sparse_fp32 = create_native_sparse_tensor((100, 100), 0.9, torch.float32)
    sparse_fp32.requires_grad_(True)
    
    # Create a dense tensor for computation
    dense = torch.randn(100, 50, dtype=torch.float32)
    
    # Forward pass
    result = torch.sparse.mm(sparse_fp32, dense)
    loss = result.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Sparse tensor gradient: shape={sparse_fp32.grad.shape}, dtype={sparse_fp32.grad.dtype}")
    print(f"Gradient is sparse: {sparse_fp32.grad.is_sparse}")

def demonstrate_common_operations():
    """Demonstrate common operations with sparse tensors in different dtypes"""
    print("\\n=== Common Operations with Different dtypes ===")
    
    # Create sparse tensors
    sparse_fp32 = create_native_sparse_tensor((500, 500), 0.95, torch.float32)
    sparse_fp16 = create_native_sparse_tensor((500, 500), 0.95, torch.float16)
    
    print("Element-wise operations:")
    
    # Addition
    sparse_sum_fp32 = sparse_fp32 + sparse_fp32
    sparse_sum_fp16 = sparse_fp16 + sparse_fp16
    print(f"  FP32 + FP32: dtype={sparse_sum_fp32.dtype}, nnz={sparse_sum_fp32._nnz()}")
    print(f"  FP16 + FP16: dtype={sparse_sum_fp16.dtype}, nnz={sparse_sum_fp16._nnz()}")
    
    # Scalar multiplication
    sparse_mul_fp32 = sparse_fp32 * 2.0
    sparse_mul_fp16 = sparse_fp16 * 2.0
    print(f"  FP32 * scalar: dtype={sparse_mul_fp32.dtype}, nnz={sparse_mul_fp32._nnz()}")
    print(f"  FP16 * scalar: dtype={sparse_mul_fp16.dtype}, nnz={sparse_mul_fp16._nnz()}")
    
    # Conversion to dense
    dense_fp32 = sparse_fp32.to_dense()
    dense_fp16 = sparse_fp16.to_dense()
    print(f"  FP32 to_dense: dtype={dense_fp32.dtype}, shape={dense_fp32.shape}")
    print(f"  FP16 to_dense: dtype={dense_fp16.dtype}, shape={dense_fp16.shape}")

def main():
    """Main function to run all native sparse examples"""
    print("Native PyTorch Sparse Tensor dtype Examples")
    print("=" * 50)
    
    demonstrate_native_sparse_creation()
    demonstrate_native_sparse_conversion()
    demonstrate_native_sparse_matmul()
    demonstrate_autocast_native_sparse()
    demonstrate_sparse_csr_dtypes()
    demonstrate_gradient_computation()
    demonstrate_common_operations()
    
    print("\\nNative sparse examples completed!")

if __name__ == "__main__":
    main()