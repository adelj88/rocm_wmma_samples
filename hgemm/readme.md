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
| Shared Memory | 0.666 | 3.22 | 0.671 | 3.20 |
| WMMA Naive | 0.516 | 4.16 | 0.508 | 4.23 |
| WMMA + Shared Memory | 0.298 | 7.21 | 0.303 | 7.09 |
| WMMA + Shared Memory + Warp Tiling | 0.393 | 5.46 | 0.415 | 5.17 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.397 | 5.41 | 0.416 | 5.16 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.187 | 11.48 | 0.214 | 10.03 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.178 | 12.06 | 0.193 | 11.13 |
| WMMA Prefetch | 0.189 | 11.36 | 0.192 | 11.18 |
| WMMA Optimized V1 | 0.170 | 12.63 | 0.180 | 11.93 |
| WMMA Optimized V2 | 0.175 | 12.27 | 0.179 | 12.00 |
| rocBLAS | 0.114 | 18.84 | 0.119 | 18.05 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.650 | 3.69 | 4.700 | 3.66 |
| WMMA Naive | 3.320 | 5.17 | 3.460 | 4.97 |
| WMMA + Shared Memory | 1.480 | 11.61 | 1.680 | 10.23 |
| WMMA + Shared Memory + Warp Tiling | 0.854 | 20.12 | 0.845 | 20.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.823 | 20.87 | 0.826 | 20.80 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.425 | 40.42 | 0.443 | 38.78 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.415 | 41.40 | 0.431 | 39.86 |
| WMMA Prefetch | 0.440 | 39.05 | 0.430 | 39.95 |
| WMMA Optimized V1 | 0.398 | 43.17 | 0.395 | 43.49 |
| WMMA Optimized V2 | 0.394 | 43.60 | 0.398 | 43.17 |
| rocBLAS | 0.336 | 51.13 | 0.326 | 52.70 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.300 | 3.79 | 43.600 | 3.15 |
| WMMA Naive | 22.000 | 6.25 | 21.600 | 6.36 |
| WMMA + Shared Memory | 10.900 | 12.61 | 12.400 | 11.08 |
| WMMA + Shared Memory + Warp Tiling | 5.740 | 23.94 | 6.520 | 21.08 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.610 | 24.50 | 6.320 | 21.75 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.380 | 57.75 | 2.440 | 56.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.330 | 58.99 | 2.390 | 57.51 |
| WMMA Prefetch | 2.340 | 58.73 | 2.380 | 57.75 |
| WMMA Optimized V1 | 2.180 | 63.05 | 2.210 | 62.19 |
| WMMA Optimized V2 | 2.150 | 63.93 | 2.130 | 64.53 |
| rocBLAS | 1.940 | 70.84 | 1.860 | 73.89 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 323.000 | 3.40 | 327.000 | 3.36 |
| WMMA Naive | 196.000 | 5.61 | 199.000 | 5.53 |
| WMMA + Shared Memory | 93.900 | 11.71 | 94.100 | 11.68 |
| WMMA + Shared Memory + Warp Tiling | 42.900 | 25.63 | 42.800 | 25.69 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.500 | 26.49 | 41.000 | 26.82 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.600 | 62.47 | 17.500 | 62.83 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 63.93 | 17.200 | 63.93 |
| WMMA Prefetch | 17.300 | 63.56 | 17.400 | 63.19 |
| WMMA Optimized V1 | 15.800 | 69.59 | 16.100 | 68.29 |
| WMMA Optimized V2 | 15.600 | 70.48 | 15.800 | 69.59 |
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
