# FP16 HGEMM with WMMA

This folder focuses on exploring Wave Matrix Multiply-Accumulate (WMMA) intrinsics in matrix-matrix multiplication.

## Features

- **Flexible Matrix Dimensions:** Supports arbitrary matrix sizes (M, N, K) beyond the basic 16x16 example
- **Multiple Implementations:**
  - Basic WMMA implementation
  - Shared memory optimized WMMA
  - Other optimizations combined with WMMA
  - Traditional shared memory implementation (for comparison)
- **Performance Benchmarking:** Built-in benchmarking capabilities for comparing different implementations
- **Correctness Verification:** CPU reference implementation for result validation

## Performance Highlights

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: No tuning has been done for different sizes.

### Square Matrix Performance Progression

The table below shows key performance points in my optimization progression:

| Implementation | 2048x2048 (TFLOPs/s) | 4096x4096 (TFLOPs/s) | 8192x8192 (TFLOPs/s) | 12288x12288 (TFLOPs/s) | 16384x16384 (TFLOPs/s) |
|----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Shared Memory  | 3.53 | 3.78 | 3.42 | 3.30 | 3.26 |
| WMMA Naive     | 5.84 | 7.19 | 5.79 | 5.55 | 3.27 |
| WMMA + Shared Memory | 11.71 | 12.40 | 11.86 | 11.87 | 11.78 |
| ... | ... | ... | ... | ... | ... |
| WMMA Optimized V3 | 47.90 | 64.15 | 77.11 | 78.06 | 76.50 |
| WMMA Optimized V4 | 50.01 | 66.32 | 80.24 | 80.38 | 80.31 |
| rocBLAS | 55.85 | 72.12 | 77.37 | 76.13 | 43.32 |

[View detailed square matrix benchmarks](docs/general.md)

### LLM-Focused Performance

The optimized WMMA implementations `wmma_opt_3`, and `wmma_opt_4` are compared against `rocBLAS` on matrix dimensions common in transformer/LLM architectures:

| Operation Type | Matrix Dimensions | `wmma_opt_3` (TFLOPs/s) | `wmma_opt_4` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_3`/`rocBLAS` | `wmma_opt_4`/`rocBLAS` |
|----------------|-------------------|-----------------|-----------------|-------------------|----------|----------|
| FFN Second Layer | m=4096, n=4096, k=16384 | 67.09 | 68.94 | 52.99 | 126.6% | 130.1% |
| Very Long Context | m=65536, n=2048, k=2048 | 78.90 | 80.68 | 61.95 | 127.4% | 130.2% |
| Attention Score | m=4096, n=2048, k=64 | 11.24 | 12.33 | 12.56 | 89.5% | 98.2% |
| Attention Score (Large Batch) | m=8192, n=4096, k=128 | 33.54 | 38.58 | 40.90 | 82.0% | 94.3% |

On average, all three optimized implementations achieve competitive performance relative to `rocBLAS` across tested LLM workloads without tuning. While `rocBLAS` maintains an edge on smaller operations like attention scores, the optimized implementations significantly outperform `rocBLAS` on FFN and long context processing tasks, with `wmma_opt_4` showing the best overall results.

[View detailed LLM benchmarks](docs/llm_focus.md)

## Verification Process

The project implements a comprehensive verification system to ensure kernel correctness and numerical stability across all implementations. The verification process includes:

### 1. Element-wise Validation
- **Comparison Method:** Each element of the GPU result matrix is compared with a CPU reference implementation
- **Adaptive Tolerance:** Different tolerances are applied based on matrix size (e.g., 0.04 for 256x256, 0.0425 for 512x512)
- **Detailed Metrics:**
  - Maximum relative error: Identifies the largest discrepancy and its location
  - Average relative error: Measures overall precision across all matrix elements
  - Number of valid comparisons: Ensures all elements are verified

### 2. Matrix Norm Validation
- **Relative Frobenius Norm Error:** Computes the difference between GPU and CPU results using matrix norms
- **Threshold-based Check:** Ensures the global error magnitude stays below acceptable limits
- **Mathematical Robustness:** Provides a single metric that captures overall numerical stability

### 3. Pattern Validation
- **Structural Similarity (SSIM):** Borrowed from image processing, this metric evaluates if the GPU result preserves the mathematical pattern of the reference
- **Threshold Check:** SSIM must be above 0.98 (98% similarity) to pass
- **Error Pattern Analysis:** Helps identify systematic issues like precision loss or algorithmic flaws

### 4. Comprehensive Reporting
The verification system provides detailed feedback for each test:
- Specific error locations and values
- Statistical summary of errors
- Pass/fail status for each validation method
- Combined overall validation status

## Known Issues

1. Some test cases are skipped for `shared` and `wmma_naive`, as there are no intentions to fix them.

## Usage

Run the executable after building:
```bash
# Assumes you're currently in /build directory
# To run unit tests
./hgemm/test

# Additionally, tests are registered with ctest
# Assumes you're currently in /build directory
cd hgemm
ctest

# To run unit benchmarks
./hgemm/bench
```

## Future Improvements

1. **WMMA HGEMM Optimization:**
   - Explore additional optimization techniques
   - Tuning best implementation for different matrix sizes
   - Add more LLM-specific matrix dimension benchmark cases
   - Further tuning for LLM-specific matrix dimensions
   - Investigate performance on future RDNA4 hardware
