#!/usr/bin/env python3
"""
Examples of using torch-sparse SparseTensor with different dtypes (fp16, fp32, bf16).
This demonstrates creating SparseTensor objects with specific dtypes, converting between dtypes,
and performing matrix multiplication operations with mixed precision.
"""

import torch
import torch.nn as nn
import time
from typing import Tuple, Dict, Any

try:
    import torch_sparse
    from torch_sparse import SparseTensor
    HAS_TORCH_SPARSE = True
    print("torch-sparse available")
except ImportError:
    HAS_TORCH_SPARSE = False
    print("torch-sparse not available")

def create_sparse_tensor_with_dtype(shape: Tuple[int, int], sparsity: float = 0.99, dtype: torch.dtype = torch.float32) -> SparseTensor:
    """Create a torch-sparse SparseTensor with specified dtype and sparsity"""
    if not HAS_TORCH_SPARSE:
        raise ImportError("torch-sparse not available")
        
    total_elements = shape[0] * shape[1]
    non_zero_elements = int(total_elements * (1 - sparsity))
    
    # Generate random indices for non-zero elements
    row = torch.randint(0, shape[0], (non_zero_elements,))
    col = torch.randint(0, shape[1], (non_zero_elements,))
    
    # Generate random values with specified dtype
    value = torch.randn(non_zero_elements, dtype=dtype)
    
    # Create torch-sparse SparseTensor
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=shape)

def demonstrate_dtype_creation():
    """Demonstrate creating SparseTensor with different dtypes"""
    print("\n=== SparseTensor dtype Creation Examples ===")
    
    if not HAS_TORCH_SPARSE:
        print("Skipping - torch-sparse not available")
        return
    
    shape = (1000, 1000)
    sparsity = 0.95
    
    # Create SparseTensor with different dtypes
    sparse_fp32 = create_sparse_tensor_with_dtype(shape, sparsity, torch.float32)
    sparse_fp16 = create_sparse_tensor_with_dtype(shape, sparsity, torch.float16)
    
    print(f"FP32 SparseTensor: shape={sparse_fp32.sparse_sizes()}, dtype={sparse_fp32.dtype()}, nnz={sparse_fp32.nnz()}")
    print(f"FP16 SparseTensor: shape={sparse_fp16.sparse_sizes()}, dtype={sparse_fp16.dtype()}, nnz={sparse_fp16.nnz()}")
    
    # Create BF16 if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        sparse_bf16 = create_sparse_tensor_with_dtype(shape, sparsity, torch.bfloat16)
        print(f"BF16 SparseTensor: shape={sparse_bf16.sparse_sizes()}, dtype={sparse_bf16.dtype()}, nnz={sparse_bf16.nnz()}")
    else:
        print("BF16 not supported on this device (requires Ampere or newer)")

def demonstrate_dtype_conversion():
    """Demonstrate converting SparseTensor between dtypes"""
    print("\n=== SparseTensor dtype Conversion Examples ===")
    
    if not HAS_TORCH_SPARSE:
        print("Skipping - torch-sparse not available")
        return
    
    # Create a base SparseTensor in FP32
    sparse_fp32 = create_sparse_tensor_with_dtype((500, 500), 0.9, torch.float32)
    print(f"Original: dtype={sparse_fp32.dtype()}, nnz={sparse_fp32.nnz()}")
    
    # Convert to FP16
    sparse_fp16 = sparse_fp32.half()
    print(f"After .half(): dtype={sparse_fp16.dtype()}, nnz={sparse_fp16.nnz()}")
    
    # Convert back to FP32
    sparse_fp32_back = sparse_fp16.float()
    print(f"After .float(): dtype={sparse_fp32_back.dtype()}, nnz={sparse_fp32_back.nnz()}")
    
    # Convert to BF16 if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        sparse_bf16 = sparse_fp32.bfloat16()
        print(f"After .bfloat16(): dtype={sparse_bf16.dtype()}, nnz={sparse_bf16.nnz()}")
    
    # Using type_as for conversion
    reference_tensor = torch.tensor([1.0], dtype=torch.float16)
    sparse_converted = sparse_fp32.type_as(reference_tensor)
    print(f"After .type_as(fp16): dtype={sparse_converted.dtype()}, nnz={sparse_converted.nnz()}")

def demonstrate_mixed_precision_matmul():
    """Demonstrate matrix multiplication with mixed precision"""
    print("\n=== Mixed Precision Matrix Multiplication Examples ===")
    
    if not HAS_TORCH_SPARSE:
        print("Skipping - torch-sparse not available")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sparse tensor and dense tensor
    sparse_shape = (1024, 2048)
    dense_shape = (2048, 512)
    sparsity = 0.95
    
    # Create tensors in different precisions
    sparse_fp32 = create_sparse_tensor_with_dtype(sparse_shape, sparsity, torch.float32)
    sparse_fp16 = create_sparse_tensor_with_dtype(sparse_shape, sparsity, torch.float16)
    
    dense_fp32 = torch.randn(dense_shape, dtype=torch.float32)
    dense_fp16 = torch.randn(dense_shape, dtype=torch.float16)
    
    if device.type == 'cuda':
        sparse_fp32 = sparse_fp32.cuda()
        sparse_fp16 = sparse_fp16.cuda()
        dense_fp32 = dense_fp32.cuda()
        dense_fp16 = dense_fp16.cuda()
    
    # FP32 @ FP32
    print("\\nFP32 @ FP32:")
    start = time.perf_counter()
    result_fp32 = sparse_fp32 @ dense_fp32
    end = time.perf_counter()
    print(f"  Result dtype: {result_fp32.dtype}, shape: {result_fp32.shape}")
    print(f"  Time: {end - start:.4f}s")
    
    # FP16 @ FP16
    print("\\nFP16 @ FP16:")
    start = time.perf_counter()
    result_fp16 = sparse_fp16 @ dense_fp16
    end = time.perf_counter()
    print(f"  Result dtype: {result_fp16.dtype}, shape: {result_fp16.shape}")
    print(f"  Time: {end - start:.4f}s")
    
    # Mixed precision: FP16 @ FP32 (automatic conversion)
    print("\\nMixed precision FP16 @ FP32:")
    start = time.perf_counter()
    result_mixed = sparse_fp16 @ dense_fp32
    end = time.perf_counter()
    print(f"  Result dtype: {result_mixed.dtype}, shape: {result_mixed.shape}")
    print(f"  Time: {end - start:.4f}s")
    
    # BF16 examples if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        sparse_bf16 = sparse_fp32.bfloat16().cuda()
        dense_bf16 = dense_fp32.bfloat16().cuda()
        
        print("\\nBF16 @ BF16:")
        start = time.perf_counter()
        result_bf16 = sparse_bf16 @ dense_bf16
        end = time.perf_counter()
        print(f"  Result dtype: {result_bf16.dtype}, shape: {result_bf16.shape}")
        print(f"  Time: {end - start:.4f}s")

def demonstrate_autocast_with_sparse():
    """Demonstrate using autocast with sparse tensors"""
    print("\n=== Autocast with Sparse Tensors ===")
    
    if not HAS_TORCH_SPARSE:
        print("Skipping - torch-sparse not available")
        return
    
    if not torch.cuda.is_available():
        print("Skipping - CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # Create tensors in FP32
    sparse_fp32 = create_sparse_tensor_with_dtype((1024, 2048), 0.95, torch.float32).cuda()
    dense_fp32 = torch.randn(2048, 512, dtype=torch.float32, device=device)
    
    print("Without autocast:")
    result_no_autocast = sparse_fp32 @ dense_fp32
    print(f"  Result dtype: {result_no_autocast.dtype}")
    
    # Note: autocast with sparse tensors may have limitations
    print("\\nWith autocast (may have limitations with sparse tensors):")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        try:
            result_autocast = sparse_fp32 @ dense_fp32
            print(f"  Result dtype: {result_autocast.dtype}")
        except Exception as e:
            print(f"  Error: {e}")
            print("  Autocast may not be fully supported with sparse tensors")

def demonstrate_memory_comparison():
    """Compare memory usage of different dtypes"""
    print("\n=== Memory Usage Comparison ===")
    
    if not HAS_TORCH_SPARSE:
        print("Skipping - torch-sparse not available")
        return
    
    shape = (2048, 2048)
    sparsity = 0.95
    
    # Create tensors with different dtypes
    sparse_fp32 = create_sparse_tensor_with_dtype(shape, sparsity, torch.float32)
    sparse_fp16 = create_sparse_tensor_with_dtype(shape, sparsity, torch.float16)
    
    def get_sparse_memory_mb(sparse_tensor):
        """Estimate memory usage of SparseTensor in MB"""
        # Memory for indices (row, col) and values
        indices_memory = sparse_tensor.nnz() * 2 * 8  # 2 int64 indices per element
        values_memory = sparse_tensor.nnz() * sparse_tensor.storage.value().element_size()
        total_memory = indices_memory + values_memory
        return total_memory / 1024 / 1024
    
    fp32_memory = get_sparse_memory_mb(sparse_fp32)
    fp16_memory = get_sparse_memory_mb(sparse_fp16)
    
    print(f"FP32 SparseTensor memory: {fp32_memory:.2f} MB")
    print(f"FP16 SparseTensor memory: {fp16_memory:.2f} MB")
    print(f"Memory savings with FP16: {(1 - fp16_memory/fp32_memory)*100:.1f}%")
    
    # Compare with dense tensor
    dense_fp32_memory = shape[0] * shape[1] * 4 / 1024 / 1024  # 4 bytes per float32
    print(f"Dense FP32 equivalent: {dense_fp32_memory:.2f} MB")
    print(f"Sparse FP32 vs Dense FP32: {(1 - fp32_memory/dense_fp32_memory)*100:.1f}% savings")

def demonstrate_limitations_and_best_practices():
    """Demonstrate limitations and best practices"""
    print("\n=== Limitations and Best Practices ===")
    
    print("1. Dtype Support:")
    print("   - FP32: Full support")
    print("   - FP16: Generally supported, may have some limitations")
    print("   - BF16: Requires Ampere+ GPUs, limited support")
    print("   - INT8: Limited support for quantization")
    
    print("\\n2. Mixed Precision Considerations:")
    print("   - Automatic dtype conversion occurs during operations")
    print("   - Result dtype typically matches the higher precision input")
    print("   - Be aware of potential precision loss")
    
    print("\\n3. Performance Considerations:")
    print("   - FP16 can be 1.5-2x faster on modern GPUs")
    print("   - BF16 offers better numerical stability than FP16")
    print("   - Memory bandwidth is often the bottleneck for sparse ops")
    
    print("\\n4. Best Practices:")
    print("   - Use FP16 for inference when possible")
    print("   - Be careful with gradient accumulation in training")
    print("   - Test numerical stability with your specific use case")
    print("   - Monitor for overflow/underflow issues")

def main():
    """Main function to run all examples"""
    print("Torch-sparse SparseTensor dtype Examples")
    print("=" * 50)
    
    if not HAS_TORCH_SPARSE:
        print("ERROR: torch-sparse not available. Please install with:")
        print("  pip install torch-sparse")
        return
    
    demonstrate_dtype_creation()
    demonstrate_dtype_conversion()
    demonstrate_mixed_precision_matmul()
    demonstrate_autocast_with_sparse()
    demonstrate_memory_comparison()
    demonstrate_limitations_and_best_practices()
    
    print("\\nExamples completed!")

if __name__ == "__main__":
    main()