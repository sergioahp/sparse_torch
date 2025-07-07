#!/usr/bin/env python3
import torch
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import os

try:
    import torch_sparse
    HAS_TORCH_SPARSE = True
except ImportError:
    HAS_TORCH_SPARSE = False

def check_dtype_support(dtype: torch.dtype, operation: str) -> bool:
    """Check if a dtype is supported for a given operation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Known unsupported combinations based on testing
    if device.type == 'cuda':
        if operation == 'pytorch_sparse':
            # PyTorch sparse CUDA operations don't support FP16/BF16
            return dtype == torch.float32
        elif operation == 'torch_sparse':
            # torch-sparse doesn't support BF16 properly
            return dtype in [torch.float32, torch.float16]
    
    # CPU might have different support - for now assume FP32 only works reliably
    return dtype == torch.float32

def create_sparse_tensor(shape: Tuple[int, int], sparsity: float, dtype: torch.dtype) -> torch.Tensor:
    """Create PyTorch sparse COO tensor with specified sparsity and dtype"""
    total_elements = shape[0] * shape[1]
    non_zero_elements = int(total_elements * (1 - sparsity))
    
    row = torch.randint(0, shape[0], (non_zero_elements,))
    col = torch.randint(0, shape[1], (non_zero_elements,))
    indices = torch.stack([row, col])
    values = torch.randn(non_zero_elements, dtype=dtype)
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)
    return sparse_tensor.coalesce()

def create_torch_sparse_tensor(shape: Tuple[int, int], sparsity: float, dtype: torch.dtype):
    """Create torch-sparse SparseTensor with specified sparsity and dtype"""
    if not HAS_TORCH_SPARSE:
        return None
        
    total_elements = shape[0] * shape[1]
    non_zero_elements = int(total_elements * (1 - sparsity))
    
    row = torch.randint(0, shape[0], (non_zero_elements,))
    col = torch.randint(0, shape[1], (non_zero_elements,))
    value = torch.randn(non_zero_elements, dtype=dtype)
    
    sparse_tensor = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=shape)
    
    # Convert to specified dtype
    if dtype == torch.float16:
        sparse_tensor = sparse_tensor.half()
    elif dtype == torch.bfloat16:
        sparse_tensor = sparse_tensor.bfloat16()
    elif dtype == torch.float32:
        sparse_tensor = sparse_tensor.float()
    
    return sparse_tensor

def benchmark_matmul(op_name: str, operation, warmup: int = 5, runs: int = 10) -> float:
    """Benchmark matrix multiplication operation"""
    # Warmup
    for _ in range(warmup):
        try:
            result = operation()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            return float('inf')  # Return inf for failed operations
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    try:
        for _ in range(runs):
            result = operation()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        return (end_time - start_time) / runs
    except Exception as e:
        return float('inf')

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different configurations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running comprehensive benchmark on {device}")
    
    # Configuration parameters
    batch_sizes = [1024, 2048, 4096, 8192]  # b
    d_model = 2048
    d_sae_multipliers = [8, 12, 16, 24]  # d_sae = 2048 * multiplier
    sparsity_levels = [0.94, 0.96, 0.98, 0.99, 0.998]  # 6% to 0.2% active
    dtypes = [torch.float32, torch.float16]
    
    # Add bfloat16 if supported
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        dtypes.append(torch.bfloat16)
    
    results = []
    total_configs = len(batch_sizes) * len(d_sae_multipliers) * len(sparsity_levels) * len(dtypes)
    config_count = 0
    
    # Track skipped configurations
    skipped_count = 0
    
    for b in batch_sizes:
        for d_sae_mult in d_sae_multipliers:
            d_sae = 2048 * d_sae_mult
            sparse_shape = (b, d_sae)  # (b, d_sae)
            dense_shape = (d_sae, d_model)  # (d_sae, d_model)
            
            for sparsity in sparsity_levels:
                active_pct = (1 - sparsity) * 100
                
                for dtype in dtypes:
                    config_count += 1
                    
                    # Check dtype support
                    pytorch_supported = check_dtype_support(dtype, 'pytorch_sparse')
                    torch_sparse_supported = check_dtype_support(dtype, 'torch_sparse') and HAS_TORCH_SPARSE
                    
                    if not pytorch_supported and not torch_sparse_supported:
                        print(f"Config {config_count}/{total_configs}: SKIPPED - "
                              f"b={b}, d_sae={d_sae}, active={active_pct:.1f}%, dtype={dtype} "
                              f"(unsupported)")
                        skipped_count += 1
                        continue
                    
                    print(f"\nConfig {config_count}/{total_configs}: "
                          f"b={b}, d_sae={d_sae}, active={active_pct:.1f}%, dtype={dtype}")
                    
                    try:
                        # Create tensors
                        sparse_tensor = create_sparse_tensor(sparse_shape, sparsity, dtype).to(device)
                        dense_tensor = torch.randn(dense_shape, dtype=dtype, device=device)
                        sparse_dense = sparse_tensor.to_dense()
                        
                        # PyTorch sparse benchmark
                        pytorch_time = float('inf')
                        if pytorch_supported:
                            def pytorch_sparse_mm():
                                return torch.sparse.mm(sparse_tensor, dense_tensor)
                            
                            pytorch_time = benchmark_matmul("pytorch_sparse", pytorch_sparse_mm)
                        else:
                            print(f"  PyTorch sparse: SKIPPED (dtype {dtype} not supported)")
                        
                        # Dense benchmark
                        def dense_mm():
                            return torch.mm(sparse_dense, dense_tensor)
                        
                        dense_time = benchmark_matmul("dense", dense_mm)
                        
                        # torch-sparse benchmark
                        torch_sparse_time = float('inf')
                        if torch_sparse_supported:
                            try:
                                torch_sparse_tensor = create_torch_sparse_tensor(sparse_shape, sparsity, dtype)
                                if device.type == 'cuda':
                                    torch_sparse_tensor = torch_sparse_tensor.cuda()
                                
                                def torch_sparse_mm():
                                    return torch_sparse_tensor @ dense_tensor
                                
                                torch_sparse_time = benchmark_matmul("torch_sparse", torch_sparse_mm)
                            except Exception as e:
                                torch_sparse_time = float('inf')
                                print(f"  torch-sparse: FAILED ({e})")
                        else:
                            print(f"  torch-sparse: SKIPPED (dtype {dtype} not supported)")
                        
                        # Calculate metrics
                        nnz = sparse_tensor._nnz()
                        actual_sparsity = 1 - (nnz / (sparse_shape[0] * sparse_shape[1]))
                        
                        sparse_memory_mb = (nnz * 2 * dtype.itemsize) / (1024 * 1024)  # indices + values
                        dense_memory_mb = (sparse_shape[0] * sparse_shape[1] * dtype.itemsize) / (1024 * 1024)
                        
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'device': str(device),
                            'batch_size': b,
                            'd_sae': d_sae,
                            'd_model': d_model,
                            'target_sparsity': sparsity,
                            'actual_sparsity': actual_sparsity,
                            'active_percentage': (1 - actual_sparsity) * 100,
                            'dtype': str(dtype),
                            'nnz': nnz,
                            'sparse_memory_mb': sparse_memory_mb,
                            'dense_memory_mb': dense_memory_mb,
                            'memory_savings_pct': (1 - sparse_memory_mb / dense_memory_mb) * 100,
                            'pytorch_sparse_time': pytorch_time,
                            'torch_sparse_time': torch_sparse_time,
                            'dense_time': dense_time,
                            'pytorch_supported': pytorch_supported,
                            'torch_sparse_supported': torch_sparse_supported,
                            'pytorch_speedup': dense_time / pytorch_time if pytorch_time != float('inf') else 0,
                            'torch_sparse_speedup': dense_time / torch_sparse_time if torch_sparse_time != float('inf') else 0,
                            'torch_sparse_vs_pytorch': pytorch_time / torch_sparse_time if torch_sparse_time != float('inf') and pytorch_time != float('inf') else 0
                        }
                        
                        results.append(result)
                        
                        # Print summary
                        pytorch_status = f"{pytorch_time:.6f}s" if pytorch_time != float('inf') else "FAILED"
                        torch_sparse_status = f"{torch_sparse_time:.6f}s" if torch_sparse_time != float('inf') else "FAILED"
                        
                        print(f"  PyTorch sparse: {pytorch_status}")
                        print(f"  torch-sparse:   {torch_sparse_status}")
                        print(f"  Dense:          {dense_time:.6f}s")
                        if pytorch_time != float('inf'):
                            print(f"  PyTorch speedup: {result['pytorch_speedup']:.2f}x")
                        if torch_sparse_time != float('inf'):
                            print(f"  torch-sparse speedup: {result['torch_sparse_speedup']:.2f}x")
                        
                    except Exception as e:
                        print(f"  Error: {e}")
                        continue
    
    print(f"\nBenchmark completed!")
    print(f"Total configurations: {config_count}")
    print(f"Skipped configurations: {skipped_count}")
    print(f"Tested configurations: {len(results)}")
    
    return results

def save_results(results: List[Dict[str, Any]], filename: str = None):
    """Save benchmark results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_fixed_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    print(f"Total configurations tested: {len(results)}")

def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics"""
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Dtype breakdown
    print(f"\nDtype Support Summary:")
    for dtype in ['torch.float32', 'torch.float16', 'torch.bfloat16']:
        dtype_results = [r for r in results if r['dtype'] == dtype]
        if dtype_results:
            pytorch_success = len([r for r in dtype_results if r['pytorch_sparse_time'] != float('inf')])
            torch_sparse_success = len([r for r in dtype_results if r['torch_sparse_time'] != float('inf')])
            print(f"  {dtype}: {len(dtype_results)} configs")
            print(f"    PyTorch sparse: {pytorch_success}/{len(dtype_results)} success")
            print(f"    torch-sparse: {torch_sparse_success}/{len(dtype_results)} success")
    
    # Best performing configurations
    valid_results = [r for r in results if r['pytorch_sparse_time'] != float('inf')]
    
    if valid_results:
        best_pytorch = max(valid_results, key=lambda x: x['pytorch_speedup'])
        print(f"\nBest PyTorch sparse speedup: {best_pytorch['pytorch_speedup']:.2f}x")
        print(f"  Config: b={best_pytorch['batch_size']}, d_sae={best_pytorch['d_sae']}, "
              f"active={best_pytorch['active_percentage']:.1f}%, dtype={best_pytorch['dtype']}")
        
        if HAS_TORCH_SPARSE:
            valid_torch_sparse = [r for r in valid_results if r['torch_sparse_time'] != float('inf')]
            if valid_torch_sparse:
                best_torch_sparse = max(valid_torch_sparse, key=lambda x: x['torch_sparse_speedup'])
                print(f"\nBest torch-sparse speedup: {best_torch_sparse['torch_sparse_speedup']:.2f}x")
                print(f"  Config: b={best_torch_sparse['batch_size']}, d_sae={best_torch_sparse['d_sae']}, "
                      f"active={best_torch_sparse['active_percentage']:.1f}%, dtype={best_torch_sparse['dtype']}")

if __name__ == "__main__":
    print("Starting FIXED comprehensive sparse tensor benchmark...")
    print(f"torch-sparse available: {HAS_TORCH_SPARSE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
    
    print(f"\nKnown limitations:")
    print(f"- PyTorch sparse CUDA: Only FP32 supported")
    print(f"- torch-sparse: FP32 and FP16 supported, BF16 fails")
    
    results = run_comprehensive_benchmark()
    save_results(results)
    print_summary(results)