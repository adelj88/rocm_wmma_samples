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

## Performance Results

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

### Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.630 | 3.41 | 0.644 | 3.33 |
| WMMA Naive | 0.467 | 4.60 | 0.473 | 4.54 |
| WMMA + Shared Memory | 0.275 | 7.81 | 0.295 | 7.28 |
| WMMA + Shared Memory + Warp Tiling | 0.367 | 5.85 | 0.389 | 5.52 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.364 | 5.90 | 0.378 | 5.68 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.161 | 13.34 | 0.177 | 12.13 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.160 | 13.42 | 0.175 | 12.27 |
| WMMA Prefetch | 0.167 | 12.86 | 0.188 | 11.42 |
| WMMA Optimized V1 | 0.156 | 13.77 | 0.163 | 13.17 |
| WMMA Optimized V2 | 0.154 | 13.94 | 0.168 | 12.78 |
| rocBLAS | 0.097 | 22.14 | 0.111 | 19.35 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.720 | 3.64 | 4.940 | 3.48 |
| WMMA Naive | 3.470 | 4.95 | 3.840 | 4.47 |
| WMMA + Shared Memory | 1.640 | 10.48 | 1.640 | 10.48 |
| WMMA + Shared Memory + Warp Tiling | 0.820 | 20.95 | 0.809 | 21.24 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.731 | 23.50 | 0.776 | 22.14 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.396 | 43.38 | 0.395 | 43.49 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.387 | 44.39 | 0.400 | 42.95 |
| WMMA Prefetch | 0.404 | 42.52 | 0.399 | 43.06 |
| WMMA Optimized V1 | 0.370 | 46.43 | 0.375 | 45.81 |
| WMMA Optimized V2 | 0.369 | 46.56 | 0.368 | 46.68 |
| rocBLAS | 0.312 | 55.06 | 0.297 | 57.84 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 37.100 | 3.70 | 41.600 | 3.30 |
| WMMA Naive | 22.400 | 6.14 | 22.400 | 6.14 |
| WMMA + Shared Memory | 10.400 | 13.22 | 12.400 | 11.08 |
| WMMA + Shared Memory + Warp Tiling | 6.320 | 21.75 | 6.580 | 20.89 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.510 | 24.94 | 6.530 | 21.05 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.690 | 51.09 | 2.370 | 57.99 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.660 | 51.67 | 2.340 | 58.73 |
| WMMA Prefetch | 2.330 | 58.99 | 2.340 | 58.73 |
| WMMA Optimized V1 | 2.180 | 63.05 | 2.150 | 63.93 |
| WMMA Optimized V2 | 2.110 | 65.14 | 2.080 | 66.08 |
| rocBLAS | 1.920 | 71.58 | 1.840 | 74.70 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 326.000 | 3.37 | 327.000 | 3.36 |
| WMMA Naive | 196.000 | 5.61 | 199.000 | 5.53 |
| WMMA + Shared Memory | 94.100 | 11.68 | 94.300 | 11.66 |
| WMMA + Shared Memory + Warp Tiling | 42.300 | 25.99 | 42.700 | 25.75 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.000 | 26.82 | 41.100 | 26.75 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 18.000 | 61.08 | 17.600 | 62.47 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 63.93 | 17.300 | 63.56 |
| WMMA Prefetch | 17.500 | 62.83 | 17.400 | 63.19 |
| WMMA Optimized V1 | 15.700 | 70.03 | 16.100 | 68.29 |
| WMMA Optimized V2 | 15.200 | 72.34 | 15.200 | 72.34 |
| rocBLAS | 14.300 | 76.89 | 14.300 | 76.89 |

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
2. rocBLAS fails some test cases (or throws an exception), so tests are skipped.

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
