#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(filename):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze_results(df):
    """Perform comprehensive analysis of benchmark results"""
    print("="*80)
    print("COMPREHENSIVE BENCHMARK ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"Total configurations tested: {len(df)}")
    print(f"Unique batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"Unique d_sae values: {sorted(df['d_sae'].unique())}")
    print(f"Unique dtypes: {df['dtype'].unique().tolist()}")
    print(f"Sparsity range: {df['actual_sparsity'].min():.3f} - {df['actual_sparsity'].max():.3f}")
    print(f"Active percentage range: {df['active_percentage'].min():.1f}% - {df['active_percentage'].max():.1f}%")
    
    # Filter out failed runs
    valid_pytorch = df[df['pytorch_sparse_time'] != float('inf')]
    valid_torch_sparse = df[df['torch_sparse_time'] != float('inf')]
    
    print(f"\nValid results:")
    print(f"PyTorch sparse: {len(valid_pytorch)}/{len(df)} ({len(valid_pytorch)/len(df)*100:.1f}%)")
    print(f"torch-sparse: {len(valid_torch_sparse)}/{len(df)} ({len(valid_torch_sparse)/len(df)*100:.1f}%)")
    
    # Performance analysis
    print(f"\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)
    
    if len(valid_pytorch) > 0:
        print(f"\nPyTorch Sparse Performance:")
        print(f"Mean speedup: {valid_pytorch['pytorch_speedup'].mean():.2f}x")
        print(f"Median speedup: {valid_pytorch['pytorch_speedup'].median():.2f}x")
        print(f"Max speedup: {valid_pytorch['pytorch_speedup'].max():.2f}x")
        print(f"Min speedup: {valid_pytorch['pytorch_speedup'].min():.2f}x")
        
        best_pytorch = valid_pytorch.loc[valid_pytorch['pytorch_speedup'].idxmax()]
        print(f"\nBest PyTorch configuration:")
        print(f"  Speedup: {best_pytorch['pytorch_speedup']:.2f}x")
        print(f"  Config: b={best_pytorch['batch_size']}, d_sae={best_pytorch['d_sae']}, "
              f"active={best_pytorch['active_percentage']:.1f}%, dtype={best_pytorch['dtype']}")
        print(f"  Time: {best_pytorch['pytorch_sparse_time']:.6f}s vs {best_pytorch['dense_time']:.6f}s")
    
    if len(valid_torch_sparse) > 0:
        print(f"\ntorch-sparse Performance:")
        print(f"Mean speedup: {valid_torch_sparse['torch_sparse_speedup'].mean():.2f}x")
        print(f"Median speedup: {valid_torch_sparse['torch_sparse_speedup'].median():.2f}x")
        print(f"Max speedup: {valid_torch_sparse['torch_sparse_speedup'].max():.2f}x")
        print(f"Min speedup: {valid_torch_sparse['torch_sparse_speedup'].min():.2f}x")
        
        best_torch_sparse = valid_torch_sparse.loc[valid_torch_sparse['torch_sparse_speedup'].idxmax()]
        print(f"\nBest torch-sparse configuration:")
        print(f"  Speedup: {best_torch_sparse['torch_sparse_speedup']:.2f}x")
        print(f"  Config: b={best_torch_sparse['batch_size']}, d_sae={best_torch_sparse['d_sae']}, "
              f"active={best_torch_sparse['active_percentage']:.1f}%, dtype={best_torch_sparse['dtype']}")
        print(f"  Time: {best_torch_sparse['torch_sparse_time']:.6f}s vs {best_torch_sparse['dense_time']:.6f}s")
    
    # Head-to-head comparison
    if len(valid_torch_sparse) > 0:
        both_valid = df[(df['pytorch_sparse_time'] != float('inf')) & 
                       (df['torch_sparse_time'] != float('inf'))]
        
        if len(both_valid) > 0:
            print(f"\n" + "="*50)
            print("HEAD-TO-HEAD COMPARISON")
            print("="*50)
            
            pytorch_wins = len(both_valid[both_valid['torch_sparse_vs_pytorch'] > 1])
            torch_sparse_wins = len(both_valid[both_valid['torch_sparse_vs_pytorch'] < 1])
            
            print(f"Configurations where PyTorch sparse wins: {pytorch_wins}/{len(both_valid)} ({pytorch_wins/len(both_valid)*100:.1f}%)")
            print(f"Configurations where torch-sparse wins: {torch_sparse_wins}/{len(both_valid)} ({torch_sparse_wins/len(both_valid)*100:.1f}%)")
            
            mean_ratio = both_valid['torch_sparse_vs_pytorch'].mean()
            print(f"Average torch-sparse vs PyTorch ratio: {mean_ratio:.2f}x")
    
    # Memory analysis
    print(f"\n" + "="*50)
    print("MEMORY ANALYSIS")
    print("="*50)
    
    print(f"Memory savings range: {df['memory_savings_pct'].min():.1f}% - {df['memory_savings_pct'].max():.1f}%")
    print(f"Mean memory savings: {df['memory_savings_pct'].mean():.1f}%")
    
    # Sparsity impact
    print(f"\n" + "="*50)
    print("SPARSITY IMPACT ANALYSIS")
    print("="*50)
    
    sparsity_groups = valid_pytorch.groupby(pd.cut(valid_pytorch['active_percentage'], bins=5))
    for name, group in sparsity_groups:
        if len(group) > 0:
            print(f"Active {name}: Mean speedup = {group['pytorch_speedup'].mean():.2f}x ({len(group)} configs)")
    
    # Dtype impact
    print(f"\n" + "="*50)
    print("DTYPE IMPACT ANALYSIS")
    print("="*50)
    
    for dtype in df['dtype'].unique():
        dtype_data = valid_pytorch[valid_pytorch['dtype'] == dtype]
        if len(dtype_data) > 0:
            print(f"{dtype}: Mean speedup = {dtype_data['pytorch_speedup'].mean():.2f}x ({len(dtype_data)} configs)")
    
    return df

def create_plots(df):
    """Create comprehensive visualizations"""
    print(f"\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter valid data
    valid_pytorch = df[df['pytorch_sparse_time'] != float('inf')]
    valid_torch_sparse = df[df['torch_sparse_time'] != float('inf')]
    both_valid = df[(df['pytorch_sparse_time'] != float('inf')) & 
                   (df['torch_sparse_time'] != float('inf'))]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Speedup vs Sparsity
    plt.subplot(3, 3, 1)
    if len(valid_pytorch) > 0:
        plt.scatter(valid_pytorch['active_percentage'], valid_pytorch['pytorch_speedup'], 
                   alpha=0.7, label='PyTorch Sparse', s=50)
    if len(valid_torch_sparse) > 0:
        plt.scatter(valid_torch_sparse['active_percentage'], valid_torch_sparse['torch_sparse_speedup'], 
                   alpha=0.7, label='torch-sparse', s=50)
    plt.xlabel('Active Percentage (%)')
    plt.ylabel('Speedup vs Dense')
    plt.title('Speedup vs Sparsity Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Speedup vs Batch Size
    plt.subplot(3, 3, 2)
    if len(valid_pytorch) > 0:
        batch_speedup = valid_pytorch.groupby('batch_size')['pytorch_speedup'].mean()
        plt.plot(batch_speedup.index, batch_speedup.values, 'o-', label='PyTorch Sparse', linewidth=2, markersize=8)
    if len(valid_torch_sparse) > 0:
        batch_speedup_ts = valid_torch_sparse.groupby('batch_size')['torch_sparse_speedup'].mean()
        plt.plot(batch_speedup_ts.index, batch_speedup_ts.values, 's-', label='torch-sparse', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Speedup vs Dense')
    plt.title('Speedup vs Batch Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Speedup vs d_sae
    plt.subplot(3, 3, 3)
    if len(valid_pytorch) > 0:
        dsae_speedup = valid_pytorch.groupby('d_sae')['pytorch_speedup'].mean()
        plt.plot(dsae_speedup.index, dsae_speedup.values, 'o-', label='PyTorch Sparse', linewidth=2, markersize=8)
    if len(valid_torch_sparse) > 0:
        dsae_speedup_ts = valid_torch_sparse.groupby('d_sae')['torch_sparse_speedup'].mean()
        plt.plot(dsae_speedup_ts.index, dsae_speedup_ts.values, 's-', label='torch-sparse', linewidth=2, markersize=8)
    plt.xlabel('d_sae Dimension')
    plt.ylabel('Mean Speedup vs Dense')
    plt.title('Speedup vs d_sae Dimension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Dtype comparison
    plt.subplot(3, 3, 4)
    dtype_data = []
    labels = []
    for dtype in df['dtype'].unique():
        pytorch_dtype = valid_pytorch[valid_pytorch['dtype'] == dtype]['pytorch_speedup']
        if len(pytorch_dtype) > 0:
            dtype_data.append(pytorch_dtype.values)
            labels.append(f'PyTorch {dtype.split(".")[-1]}')
        
        torch_sparse_dtype = valid_torch_sparse[valid_torch_sparse['dtype'] == dtype]['torch_sparse_speedup']
        if len(torch_sparse_dtype) > 0:
            dtype_data.append(torch_sparse_dtype.values)
            labels.append(f'torch-sparse {dtype.split(".")[-1]}')
    
    if dtype_data:
        plt.boxplot(dtype_data, labels=labels)
        plt.ylabel('Speedup vs Dense')
        plt.title('Speedup Distribution by Dtype')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 5. Head-to-head comparison
    plt.subplot(3, 3, 5)
    if len(both_valid) > 0:
        plt.scatter(both_valid['pytorch_speedup'], both_valid['torch_sparse_speedup'], 
                   c=both_valid['active_percentage'], cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(label='Active %')
        
        # Add diagonal line
        max_speedup = max(both_valid['pytorch_speedup'].max(), both_valid['torch_sparse_speedup'].max())
        plt.plot([0, max_speedup], [0, max_speedup], 'r--', alpha=0.7, label='Equal Performance')
        
        plt.xlabel('PyTorch Sparse Speedup')
        plt.ylabel('torch-sparse Speedup')
        plt.title('Head-to-Head Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Memory savings vs Performance
    plt.subplot(3, 3, 6)
    if len(valid_pytorch) > 0:
        plt.scatter(valid_pytorch['memory_savings_pct'], valid_pytorch['pytorch_speedup'], 
                   alpha=0.7, label='PyTorch Sparse', s=50)
    if len(valid_torch_sparse) > 0:
        plt.scatter(valid_torch_sparse['memory_savings_pct'], valid_torch_sparse['torch_sparse_speedup'], 
                   alpha=0.7, label='torch-sparse', s=50)
    plt.xlabel('Memory Savings (%)')
    plt.ylabel('Speedup vs Dense')
    plt.title('Memory Savings vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Absolute timing comparison
    plt.subplot(3, 3, 7)
    if len(both_valid) > 0:
        plt.scatter(both_valid['pytorch_sparse_time'], both_valid['torch_sparse_time'], 
                   c=both_valid['batch_size'], cmap='plasma', alpha=0.7, s=50)
        plt.colorbar(label='Batch Size')
        
        # Add diagonal line
        max_time = max(both_valid['pytorch_sparse_time'].max(), both_valid['torch_sparse_time'].max())
        plt.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, label='Equal Time')
        
        plt.xlabel('PyTorch Sparse Time (s)')
        plt.ylabel('torch-sparse Time (s)')
        plt.title('Absolute Timing Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.loglog()
    
    # 8. Heatmap of configurations
    plt.subplot(3, 3, 8)
    if len(valid_pytorch) > 0:
        # Create pivot table for heatmap
        heatmap_data = valid_pytorch.pivot_table(
            values='pytorch_speedup', 
            index='active_percentage', 
            columns='batch_size', 
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'PyTorch Speedup'})
        plt.title('PyTorch Speedup Heatmap\n(Active % vs Batch Size)')
        plt.ylabel('Active Percentage (%)')
        plt.xlabel('Batch Size')
    
    # 9. Performance trends by dimension
    plt.subplot(3, 3, 9)
    if len(valid_pytorch) > 0:
        # Calculate total FLOPs proxy (batch_size * d_sae * d_model)
        valid_pytorch['total_ops'] = valid_pytorch['batch_size'] * valid_pytorch['d_sae'] * valid_pytorch['d_model']
        
        plt.scatter(valid_pytorch['total_ops'], valid_pytorch['pytorch_speedup'], 
                   alpha=0.7, label='PyTorch Sparse', s=50)
        
        if len(valid_torch_sparse) > 0:
            valid_torch_sparse['total_ops'] = valid_torch_sparse['batch_size'] * valid_torch_sparse['d_sae'] * valid_torch_sparse['d_model']
            plt.scatter(valid_torch_sparse['total_ops'], valid_torch_sparse['torch_sparse_speedup'], 
                       alpha=0.7, label='torch-sparse', s=50)
        
        plt.xlabel('Total Operations (B×d_sae×d_model)')
        plt.ylabel('Speedup vs Dense')
        plt.title('Speedup vs Problem Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.semilogx()
    
    plt.tight_layout()
    plt.savefig('benchmark_analysis.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'benchmark_analysis.png'")
    
    plt.show()

def main():
    # Find the most recent results file
    json_files = list(Path('.').glob('benchmark_results_*.json'))
    if not json_files:
        print("No benchmark results files found!")
        return
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    # Load and analyze data
    df = load_results(latest_file)
    analyzed_df = analyze_results(df)
    
    # Create visualizations
    create_plots(analyzed_df)
    
    # Save processed data
    output_file = latest_file.stem + '_processed.csv'
    analyzed_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved as: {output_file}")

if __name__ == "__main__":
    main()