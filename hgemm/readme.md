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
| Shared Memory | 0.578 | 3.76 | 0.579 | 3.75 |
| WMMA Naive | 0.413 | 5.26 | 0.423 | 5.14 |
| WMMA + Shared Memory | 0.219 | 9.93 | 0.220 | 9.87 |
| WMMA + Shared Memory + Warp Tiling | 0.412 | 5.28 | 0.342 | 6.36 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.406 | 5.36 | 0.337 | 6.45 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.118 | 18.42 | 0.119 | 18.27 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.120 | 18.12 | 0.119 | 18.27 |
| rocBLAS | 0.060 | 36.29 | 0.057 | 38.19 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.693 | 3.67 | 4.685 | 3.67 |
| WMMA Naive | 3.474 | 4.96 | 3.251 | 5.29 |
| WMMA + Shared Memory | 1.586 | 10.84 | 1.393 | 12.34 |
| WMMA + Shared Memory + Warp Tiling | 0.913 | 18.83 | 0.757 | 22.72 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.873 | 19.69 | 0.726 | 23.68 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.328 | 52.42 | 0.343 | 50.14 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.336 | 51.14 | 0.344 | 50.00 |
| rocBLAS | 0.248 | 69.34 | 0.250 | 68.76 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.912 | 3.73 | 37.429 | 3.67 |
| WMMA Naive | 23.477 | 5.86 | 21.456 | 6.40 |
| WMMA + Shared Memory | 11.036 | 12.47 | 11.030 | 12.48 |
| WMMA + Shared Memory + Warp Tiling | 6.213 | 22.16 | 6.249 | 22.00 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.395 | 25.51 | 6.103 | 22.54 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.302 | 59.78 | 2.305 | 59.69 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.176 | 63.24 | 2.203 | 62.44 |
| rocBLAS | 1.794 | 76.67 | 1.805 | 76.22 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 323.825 | 3.40 | 329.053 | 3.34 |
| WMMA Naive | 196.329 | 5.60 | 200.475 | 5.49 |
| WMMA + Shared Memory | 93.942 | 11.71 | 94.244 | 11.67 |
| WMMA + Shared Memory + Warp Tiling | 42.432 | 25.92 | 42.469 | 25.90 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.069 | 26.78 | 40.773 | 26.98 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.530 | 62.75 | 17.542 | 62.71 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 16.808 | 65.45 | 16.872 | 65.20 |
| rocBLAS | 14.174 | 77.61 | 14.245 | 77.23 |

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
