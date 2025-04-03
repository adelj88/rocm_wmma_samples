# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available.

## Objectives
This project aims to:
1. Provide a simple example of HIP programming and WMMA usage for GPU-accelerated computation
2. Extend beyond the fixed-size example in the GPUOpen tutorial by supporting arbitrary matrix dimensions (M, N, K)
3. Enhance understanding of the WMMA intrinsic's mechanics, especially around data loading and storing

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
| Shared Memory  | 3.65 | 3.76 | 3.39 | 3.37 | 3.35 |
| WMMA Naive     | 5.30 | 6.28 | 5.57 | 5.39 | 3.29 |
| WMMA + Shared Memory | 11.40 | 13.14 | 11.68 | 11.73 | 11.63 |
| ... | ... | ... | ... | ... | ... |
| WMMA Optimized V2 | 47.88 | 63.09 | 76.91 | 74.66 | 76.22 |
| WMMA Optimized V3 | 47.56 | 63.54 | 76.90 | 77.24 | 76.21 |
| rocBLAS | 51.39 | 70.94 | 76.99 | 76.05 | 43.53 |

[View detailed square matrix benchmarks](docs/general.md)

### LLM-Focused Performance

The optimized WMMA implementations `wmma_opt_2` and `wmma_opt_3` are compared against `rocBLAS` on matrix dimensions common in transformer/LLM architectures:

| Operation Type | Matrix Dimensions | `wmma_opt_2` (TFLOPs/s) | `wmma_opt_3` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` | `wmma_opt_3`/`rocBLAS` |
|----------------|-------------------|-----------------|-----------------|-------------------|----------|----------|
| FFN Second Layer | m=4096, n=4096, k=16384 | 67.32 | 67.15 | 54.03 | 124.6% | 124.3% |
| Very Long Context | m=65536, n=2048, k=2048 | 69.14 | 78.97 | 61.51 | 112.4% | 128.4% |
| Attention Score | m=4096, n=2048, k=64 | 11.23 | 11.54 | 12.77 | 87.9% | 90.3% |
| Attention Score (Large Batch) | m=8192, n=4096, k=128 | 32.25 | 33.04 | 42.47 | 75.9% | 77.8% |

On average, both `wmma_opt_2` and `wmma_opt_3` achieve competitive performance relative to `rocBLAS` across tested LLM workloads without tuning, with `wmma_opt_3` showing particularly strong results on FFN and long context processing tasks.

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
- **Threshold Check:** SSIM must be above 0.95 (95% similarity) to pass
- **Error Pattern Analysis:** Helps identify systematic issues like precision loss or algorithmic flaws

### 4. Comprehensive Reporting
The verification system provides detailed feedback for each test:
- Specific error locations and values
- Statistical summary of errors
- Pass/fail status for each validation method
- Combined overall validation status

## Known Issues

1. Some test cases are skipped for `shared` and `wmma_naive`, as there are no intentions to fix them.
2. `rocBLAS` fails some test cases (or throws an exception), so tests are skipped.

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
   - Tuning `wmma_opt_2` for different matrix sizes
   - Add more LLM-specific matrix dimension benchmark cases
   - Further tuning for LLM-specific matrix dimensions
   - Investigate performance on future RDNA4 hardware
