# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available.

## Objectives
This project aims to:
1. Provide a simple example of HIP programming and WMMA usage for GPU-accelerated computation
2. Extend beyond the fixed-size example in the GPUOpen tutorial by supporting arbitrary power-of-two matrix dimensions (M, N, K)
3. Enhance understanding of the WMMA intrinsic's mechanics, especially around data loading and storing

## Features

- **Flexible Matrix Dimensions:** Supports arbitrary power-of-two matrix sizes (M, N, K) beyond the basic 16x16 example
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
| Shared Memory | 0.572 | 3.80 | 0.576 | 3.78 |
| WMMA Naive | 0.410 | 5.31 | 0.425 | 5.12 |
| WMMA + Shared Memory | 0.215 | 10.10 | 0.222 | 9.81 |
| WMMA + Shared Memory + Warp Tiling | 0.406 | 5.36 | 0.341 | 6.37 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.403 | 5.39 | 0.335 | 6.50 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.118 | 18.42 | 0.122 | 17.87 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.116 | 18.78 | 0.118 | 18.45 |
| WMMA Prefetch | 0.113 | 19.25 | 0.124 | 17.52 |
| WMMA Optimized V1 | 0.109 | 19.98 | 0.114 | 19.04 |
| rocBLAS | 0.059 | 36.87 | 0.052 | 41.74 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.710 | 3.65 | 4.728 | 3.64 |
| WMMA Naive | 3.544 | 4.86 | 3.227 | 5.33 |
| WMMA + Shared Memory | 1.574 | 10.92 | 1.385 | 12.42 |
| WMMA + Shared Memory + Warp Tiling | 0.917 | 18.76 | 0.769 | 22.37 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.892 | 19.29 | 0.731 | 23.52 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.331 | 52.00 | 0.339 | 50.76 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.332 | 51.85 | 0.354 | 48.59 |
| WMMA Prefetch | 0.344 | 49.96 | 0.362 | 47.52 |
| WMMA Optimized V1 | 0.323 | 53.28 | 0.340 | 50.57 |
| rocBLAS | 0.248 | 69.36 | 0.243 | 70.76 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.941 | 3.72 | 37.337 | 3.69 |
| WMMA Naive | 23.408 | 5.88 | 21.522 | 6.39 |
| WMMA + Shared Memory | 10.296 | 13.37 | 10.958 | 12.56 |
| WMMA + Shared Memory + Warp Tiling | 6.203 | 22.18 | 6.197 | 22.21 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.066 | 22.69 | 6.076 | 22.65 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.238 | 61.50 | 2.309 | 59.59 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.198 | 62.65 | 2.255 | 61.01 |
| WMMA Prefetch | 2.234 | 61.62 | 2.301 | 59.80 |
| WMMA Optimized V1 | 2.094 | 65.71 | 2.133 | 64.51 |
| rocBLAS | 1.749 | 78.65 | 1.740 | 79.04 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 322.962 | 3.40 | 328.631 | 3.35 |
| WMMA Naive | 195.017 | 5.64 | 199.790 | 5.51 |
| WMMA + Shared Memory | 93.346 | 11.78 | 94.057 | 11.70 |
| WMMA + Shared Memory + Warp Tiling | 42.220 | 26.06 | 42.201 | 26.07 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 40.874 | 26.91 | 40.517 | 27.15 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.475 | 63.06 | 17.471 | 63.07 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.063 | 64.58 | 17.116 | 64.37 |
| WMMA Prefetch | 17.116 | 64.38 | 17.308 | 63.67 |
| WMMA Optimized V1 | 15.726 | 70.07 | 16.093 | 68.48 |
| rocBLAS | 14.110 | 78.09 | 14.154 | 77.85 |

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

### 5. Size-Based Testing
- Small matrices (256x256, 512x512) undergo full verification with all metrics
- Larger matrices (1024+ dimensions) focus on performance benchmarking after correctness is established

This multi-faceted approach ensures that kernel optimizations maintain numerical correctness while improving performance. As shown in the test results, all implementations achieve high accuracy with maximum relative errors under 1% and SSIM values above 0.98, indicating reliable computation regardless of the optimization techniques applied.

## Known Issues

1. Current implementations are limited to the work tile configurations (e.g. if a tile of 256x256x16 is used, it will only work for dimensions that are multiples of 256).

## Usage

Run the executable after building:
```bash
# Assumes you're currently in /build directory
./hgemm/hgemm
```

### Customizing Tests

- Edit `.verify_sizes` in main.cpp to add specific matrix sizes for correctness validation
- Edit `.benchmark_sizes` to specify sizes for performance testing

## Future Improvements

1. **WMMA HGEMM Optimization:**
   - Explore additional optimization techniques
