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

### Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.644 | 3.33 | 0.720 | 2.98 |
| WMMA Naive | 0.506 | 4.24 | 0.632 | 3.40 |
| WMMA + Shared Memory | 0.305 | 7.04 | 0.346 | 6.21 |
| WMMA + Shared Memory + Warp Tiling | 0.397 | 5.41 | 0.501 | 4.29 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.402 | 5.34 | 0.505 | 4.25 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.189 | 11.36 | 0.277 | 7.75 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.185 | 11.61 | 0.279 | 7.70 |
| WMMA Prefetch | 0.196 | 10.96 | 0.321 | 6.69 |
| WMMA Optimized V1 | 0.181 | 11.86 | 0.242 | 8.87 |
| WMMA Optimized V2 | 0.174 | 12.34 | 0.229 | 9.38 |
| rocBLAS | 0.117 | 18.35 | 0.130 | 16.52 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.830 | 3.56 | 4.770 | 3.60 |
| WMMA Naive | 3.290 | 5.22 | 3.580 | 4.80 |
| WMMA + Shared Memory | 1.530 | 11.23 | 1.720 | 9.99 |
| WMMA + Shared Memory + Warp Tiling | 0.855 | 20.09 | 0.867 | 19.82 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.816 | 21.05 | 0.793 | 21.66 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.418 | 41.10 | 0.497 | 34.57 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.409 | 42.00 | 0.565 | 30.41 |
| WMMA Prefetch | 0.426 | 40.33 | 0.574 | 29.93 |
| WMMA Optimized V1 | 0.400 | 42.95 | 0.464 | 37.03 |
| WMMA Optimized V2 | 0.391 | 43.94 | 0.428 | 40.14 |
| rocBLAS | 0.341 | 50.38 | 0.332 | 51.75 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 37.000 | 3.71 | 43.300 | 3.17 |
| WMMA Naive | 22.000 | 6.25 | 22.000 | 6.25 |
| WMMA + Shared Memory | 10.700 | 12.84 | 12.000 | 11.45 |
| WMMA + Shared Memory + Warp Tiling | 6.480 | 21.21 | 5.850 | 23.49 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.250 | 21.99 | 5.740 | 23.94 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.370 | 57.99 | 2.430 | 56.56 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.330 | 58.99 | 2.480 | 55.42 |
| WMMA Prefetch | 2.340 | 58.73 | 2.400 | 57.27 |
| WMMA Optimized V1 | 2.160 | 63.63 | 2.200 | 62.47 |
| WMMA Optimized V2 | 2.140 | 64.22 | 2.230 | 61.63 |
| rocBLAS | 1.940 | 70.84 | 1.860 | 73.89 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 324.000 | 3.39 | 327.000 | 3.36 |
| WMMA Naive | 196.000 | 5.61 | 200.000 | 5.50 |
| WMMA + Shared Memory | 93.700 | 11.73 | 93.800 | 11.72 |
| WMMA + Shared Memory + Warp Tiling | 42.300 | 25.99 | 42.400 | 25.93 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 40.900 | 26.88 | 40.500 | 27.15 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.600 | 62.47 | 17.500 | 62.83 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 63.93 | 17.100 | 64.30 |
| WMMA Prefetch | 17.400 | 63.19 | 17.300 | 63.56 |
| WMMA Optimized V1 | 15.800 | 69.59 | 15.900 | 69.15 |
| WMMA Optimized V2 | 15.400 | 71.40 | 15.600 | 70.48 |
| rocBLAS | 14.300 | 76.89 | 14.200 | 77.43 |

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
