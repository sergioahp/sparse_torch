#!/usr/bin/env python3
import torch
import traceback
import logging
from datetime import datetime

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'debug_inf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

try:
    import torch_sparse
    HAS_TORCH_SPARSE = True
    logging.info("torch-sparse imported successfully")
except ImportError:
    HAS_TORCH_SPARSE = False
    logging.warning("torch-sparse not available")

def test_sparse_operation(dtype, sparsity, shape, operation_name):
    """Test a specific sparse operation configuration"""
    test_name = f"{operation_name} with {dtype}, sparsity={sparsity:.1%}, shape={shape}"
    print(f"\n=== Testing {test_name} ===")
    logging.info(f"Starting test: {test_name}")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create sparse tensor
        total_elements = shape[0] * shape[1]
        non_zero_elements = int(total_elements * (1 - sparsity))
        
        row = torch.randint(0, shape[0], (non_zero_elements,))
        col = torch.randint(0, shape[1], (non_zero_elements,))
        indices = torch.stack([row, col])
        values = torch.randn(non_zero_elements, dtype=dtype)
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype).to(device)
        sparse_tensor = sparse_tensor.coalesce()
        
        print(f"✓ Created sparse tensor: {sparse_tensor.shape}, nnz={sparse_tensor._nnz()}")
        logging.info(f"Created sparse tensor: {sparse_tensor.shape}, nnz={sparse_tensor._nnz()}, dtype={sparse_tensor.dtype}")
        
        # Create dense tensor for multiplication
        dense_shape = (shape[1], 2048)  # (d_sae, d_model)
        dense_tensor = torch.randn(dense_shape, dtype=dtype, device=device)
        
        print(f"✓ Created dense tensor: {dense_tensor.shape}")
        logging.info(f"Created dense tensor: {dense_tensor.shape}, dtype={dense_tensor.dtype}")
        
        # Test PyTorch sparse operation
        if operation_name == "pytorch_sparse":
            print("Testing torch.sparse.mm...")
            logging.info("About to call torch.sparse.mm")
            result = torch.sparse.mm(sparse_tensor, dense_tensor)
            print(f"✓ PyTorch sparse mm succeeded: {result.shape}")
            logging.info(f"PyTorch sparse mm succeeded: {result.shape}, result_dtype={result.dtype}")
            
        # Test torch-sparse operation
        elif operation_name == "torch_sparse" and HAS_TORCH_SPARSE:
            print("Testing torch-sparse...")
            logging.info("Creating torch-sparse tensor")
            
            # Create torch-sparse tensor
            ts_tensor = torch_sparse.SparseTensor(
                row=sparse_tensor._indices()[0],
                col=sparse_tensor._indices()[1], 
                value=sparse_tensor._values(),
                sparse_sizes=shape
            )
            
            logging.info(f"Created torch-sparse tensor, original dtype: {ts_tensor.dtype()}")
            
            # Convert to specified dtype
            if dtype == torch.float16:
                logging.info("Converting to half precision")
                ts_tensor = ts_tensor.half()
            elif dtype == torch.bfloat16:
                logging.info("Converting to bfloat16")
                ts_tensor = ts_tensor.bfloat16()
            elif dtype == torch.float32:
                logging.info("Converting to float32")
                ts_tensor = ts_tensor.float()
                
            if device.type == 'cuda':
                logging.info("Moving torch-sparse tensor to CUDA")
                ts_tensor = ts_tensor.cuda()
                
            print(f"✓ Created torch-sparse tensor: {ts_tensor.sparse_sizes()}")
            logging.info(f"Final torch-sparse tensor dtype: {ts_tensor.dtype()}")
            
            logging.info("About to perform torch-sparse matrix multiplication")
            result = ts_tensor @ dense_tensor
            print(f"✓ torch-sparse @ succeeded: {result.shape}")
            logging.info(f"torch-sparse @ succeeded: {result.shape}, result_dtype={result.dtype}")
            
        logging.info(f"Test completed successfully: {test_name}")
        return True
        
    except Exception as e:
        error_msg = f"FAILED: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        logging.error(f"Test failed: {test_name}")
        logging.error(f"Error: {error_msg}")
        logging.error(f"Full traceback:")
        
        # Log full traceback
        import io
        import sys
        f = io.StringIO()
        traceback.print_exc(file=f)
        logging.error(f.getvalue())
        
        print(f"Traceback:")
        traceback.print_exc()
        return False

def main():
    """Test various configurations to identify failure patterns"""
    print("🔍 Debugging sparse tensor inf issues...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"torch-sparse available: {HAS_TORCH_SPARSE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
    
    # Test configurations that were failing
    test_configs = [
        # (dtype, sparsity, shape, operation)
        (torch.float32, 0.998, (8192, 49152), "pytorch_sparse"),
        (torch.float16, 0.998, (8192, 49152), "pytorch_sparse"),
        (torch.bfloat16, 0.998, (8192, 49152), "pytorch_sparse"),
        
        (torch.float32, 0.998, (8192, 49152), "torch_sparse"),
        (torch.float16, 0.998, (8192, 49152), "torch_sparse"),
        (torch.bfloat16, 0.998, (8192, 49152), "torch_sparse"),
        
        # Test smaller sizes
        (torch.float16, 0.99, (1024, 16384), "pytorch_sparse"),
        (torch.bfloat16, 0.99, (1024, 16384), "pytorch_sparse"),
        
        # Test different sparsity levels
        (torch.float16, 0.94, (2048, 24576), "pytorch_sparse"),
        (torch.float16, 0.98, (2048, 24576), "pytorch_sparse"),
    ]
    
    results = []
    
    for dtype, sparsity, shape, operation in test_configs:
        success = test_sparse_operation(dtype, sparsity, shape, operation)
        results.append((dtype, sparsity, shape, operation, success))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for dtype, sparsity, shape, operation, success in results:
        status = "✓" if success else "❌"
        print(f"{status} {operation} | {str(dtype).split('.')[-1]} | sparsity={sparsity:.1%} | {shape}")
    
    # Analysis
    print("\n" + "="*50)
    print("FAILURE PATTERN ANALYSIS")
    print("="*50)
    
    pytorch_fp16_failures = sum(1 for d, s, sh, op, succ in results 
                               if op == "pytorch_sparse" and "float16" in str(d) and not succ)
    pytorch_bf16_failures = sum(1 for d, s, sh, op, succ in results 
                               if op == "pytorch_sparse" and "bfloat16" in str(d) and not succ)
    
    print(f"PyTorch sparse FP16 failures: {pytorch_fp16_failures}")
    print(f"PyTorch sparse BF16 failures: {pytorch_bf16_failures}")
    
    if HAS_TORCH_SPARSE:
        ts_fp16_failures = sum(1 for d, s, sh, op, succ in results 
                              if op == "torch_sparse" and "float16" in str(d) and not succ)
        ts_bf16_failures = sum(1 for d, s, sh, op, succ in results 
                              if op == "torch_sparse" and "bfloat16" in str(d) and not succ)
        
        print(f"torch-sparse FP16 failures: {ts_fp16_failures}")
        print(f"torch-sparse BF16 failures: {ts_bf16_failures}")

if __name__ == "__main__":
    main()