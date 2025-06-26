# FP16 HGEMM with WMMA

This folder focuses on exploring Wave Matrix Multiply-Accumulate (WMMA) intrinsics in matrix-matrix multiplication.

## Features

- **Flexible Matrix Dimensions:** Supports arbitrary matrix sizes (M, N, K) beyond the basic 16x16 example
- **Multiple Implementations:**
  - Basic WMMA implementation
  - Shared memory optimized WMMA
  - Five generations of optimized WMMA implementations (V1-V5)
  - Traditional shared memory implementation (for comparison)
- **Performance Benchmarking:** Built-in benchmarking capabilities for comparing different implementations
- **Correctness Verification:** CPU reference implementation for result validation

## Performance Highlights

Performance measured on AMD Radeon RX 7900 GRE on Windows (HIP SDK 6.2.4) and WSL2 (Ubuntu 24.04.1 LTS, ROCm 6.4.1). All implementations use half precision (FP16).

Note: No tuning has been done for different sizes.

### Square Matrix Performance Progression

The table below shows key performance points in my optimization progression:

| Implementation | 2048x2048 (TFLOPs/s) | 4096x4096 (TFLOPs/s) | 8192x8192 (TFLOPs/s) |
|----------------|---------------------|---------------------|---------------------|
| Shared Memory  | 3.68 | 3.80 | 3.55 |
| WMMA Naive     | 5.39 | 6.89 | 5.77 |
| WMMA + Shared Memory | 10.75 | 12.06 | 11.74 |
| ... | ... | ... | ... |
| WMMA Optimized V3 | 50.76 | 55.51 | 73.75 |
| WMMA Optimized V4 | 52.78 | 58.81 | 76.37 |
| WMMA Optimized V5 | 52.18 | 69.49 | 76.36 |
| rocBLAS | 55.96 | 66.75 | 75.13 |

**Key Achievements:**
- **~22x speedup** from baseline shared memory to best optimized version
- **WMMA Optimized V5** achieves 94.2% of rocBLAS performance on average across LLM workloads
- Performance now matches or exceeds rocBLAS on many transformer-specific matrix shapes

[View detailed square matrix benchmarks](docs/general.md)

### LLM-Focused Performance

The optimized WMMA implementations `wmma_opt_3`, `wmma_opt_4`, and `wmma_opt_5` are compared against `rocBLAS` on matrix dimensions common in transformer/LLM architectures:

| Operation Type | Matrix Dimensions | `wmma_opt_3` (TFLOPs/s) | `wmma_opt_4` (TFLOPs/s) | `wmma_opt_5` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | Best/`rocBLAS` |
|----------------|-------------------|-----------------|-----------------|-----------------|-------------------|----------|
| QKV Projection | m=4096, n=4096, k=1024 | 55.61 | 58.63 | 66.35 | 70.41 | 94.2% |
| QKV Projection (Large Batch) | m=8192, n=8192, k=1024 | 69.57 | 73.62 | 74.28 | 74.87 | 99.2% |
| FFN First Layer | m=4096, n=16384, k=4096 | 74.93 | 78.35 | 76.78 | 76.56 | 102.3% |
| FFN Second Layer | m=4096, n=4096, k=16384 | 66.52 | 68.69 | 74.41 | 53.73 | 138.5% |
| Model with 5120 Hidden Dim | m=4096, n=5120, k=5120 | 81.72 | 84.80 | 84.00 | 75.71 | 112.0% |
| Long Context Processing | m=32768, n=4096, k=4096 | 76.84 | 79.97 | 80.08 | 76.78 | 104.3% |
| Very Long Context | m=65536, n=2048, k=2048 | 73.79 | 77.32 | 77.39 | 61.51 | 125.8% |

**Performance Summary:**
- **Competitive with rocBLAS:** Achieving 90-99% performance on most workloads
- **Exceeding rocBLAS:** Up to 38.5% faster on FFN layers and long context processing
- **Best Overall:** `wmma_opt_5` shows the most consistent performance across different workload types
- **No Tuning Required:** Results achieved without kernel parameter tuning for specific matrix sizes

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

### 2. Pattern Validation
- **Structural Similarity (SSIM):** Borrowed from image processing, this metric evaluates if the GPU result preserves the mathematical pattern of the reference
- **Threshold Check:** SSIM must be above 0.98 (98% similarity) to pass
- **Error Pattern Analysis:** Helps identify systematic issues like precision loss or algorithmic flaws

### 3. Comprehensive Reporting
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
   - Explore additional optimization techniques beyond V5
   - Investigate performance on future RDNA4 hardware
