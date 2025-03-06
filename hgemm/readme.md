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
| Shared Memory | 0.581 | 3.74 | 0.581 | 3.74 |
| WMMA Naive | 0.421 | 5.16 | 0.426 | 5.10 |
| WMMA + Shared Memory | 0.222 | 9.78 | 0.225 | 9.65 |
| WMMA + Shared Memory + Warp Tiling | 0.416 | 5.23 | 0.310 | 7.01 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.417 | 5.22 | 0.307 | 7.08 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.138 | 15.75 | 0.128 | 16.96 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.141 | 15.42 | 0.129 | 16.87 |
| rocBLAS | 0.073 | 29.84 | 0.054 | 40.34 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.702 | 3.66 | 4.663 | 3.68 |
| WMMA Naive | 3.501 | 4.92 | 3.228 | 5.32 |
| WMMA + Shared Memory | 1.584 | 10.86 | 1.401 | 12.27 |
| WMMA + Shared Memory + Warp Tiling | 0.927 | 18.54 | 0.761 | 22.60 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.886 | 19.41 | 0.668 | 25.73 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.360 | 47.73 | 0.357 | 48.20 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.350 | 49.21 | 0.355 | 48.46 |
| rocBLAS | 0.263 | 65.36 | 0.246 | 69.92 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 39.775 | 3.46 | 37.274 | 3.69 |
| WMMA Naive | 22.415 | 6.14 | 21.442 | 6.41 |
| WMMA + Shared Memory | 11.104 | 12.40 | 11.082 | 12.41 |
| WMMA + Shared Memory + Warp Tiling | 5.572 | 24.70 | 6.281 | 21.89 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.445 | 25.27 | 5.436 | 25.30 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.373 | 57.98 | 2.403 | 57.26 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.384 | 57.71 | 2.391 | 57.53 |
| rocBLAS | 1.763 | 78.04 | 1.752 | 78.44 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 325.142 | 3.38 | 328.722 | 3.35 |
| WMMA Naive | 196.534 | 5.60 | 198.914 | 5.53 |
| WMMA + Shared Memory | 93.844 | 11.72 | 94.334 | 11.66 |
| WMMA + Shared Memory + Warp Tiling | 42.497 | 25.88 | 42.562 | 25.84 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.151 | 26.73 | 40.816 | 26.95 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 18.912 | 58.17 | 19.017 | 57.84 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 18.378 | 59.86 | 18.455 | 59.61 |
| rocBLAS | 14.197 | 77.48 | 14.257 | 77.16 |

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
