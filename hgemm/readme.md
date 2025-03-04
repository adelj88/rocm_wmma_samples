# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available. For production-grade GPU matrix multiplication, it is highly recommended to use [rocWMMA](https://github.com/ROCm/rocWMMA), which provides a robust and optimized abstraction over the WMMA functionality.

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
| Shared Memory | 0.575 | 3.75 | 0.579 | 3.73 |
| WMMA Naive | 0.408 | 5.30 | 0.421 | 5.13 |
| WMMA + Shared Memory | 0.186 | 11.61 | 0.203 | 10.64 |
| WMMA + Shared Memory + Warp Tiling | 0.261 | 8.28 | 0.284 | 7.60 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.207 | 10.45 | 0.257 | 8.42 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.079 | 27.32 | 0.120 | 18.03 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.064 | 33.90 | 0.076 | 28.59 |
| WMMA Prefetch | 0.063 | 34.41 | 0.076 | 28.45 |
| rocWMMA | 0.077 | 28.07 | 0.269 | 8.04 |
| rocBLAS | 0.046 | 46.91 | 0.048 | 45.00 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.670 | 3.68 | 4.670 | 3.68 |
| WMMA Naive | 3.277 | 5.25 | 3.066 | 5.61 |
| WMMA + Shared Memory | 1.357 | 12.68 | 1.350 | 12.74 |
| WMMA + Shared Memory + Warp Tiling | 0.960 | 17.91 | 0.852 | 20.17 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.943 | 18.23 | 0.981 | 17.51 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.391 | 44.00 | 0.411 | 41.94 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.345 | 49.85 | 0.343 | 50.14 |
| WMMA Prefetch | 0.337 | 51.06 | 0.352 | 48.86 |
| rocWMMA | 0.416 | 41.36 | 3.557 | 4.84 |
| rocBLAS | 0.254 | 67.81 | 0.240 | 71.68 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.408 | 3.78 | 36.513 | 3.77 |
| WMMA Naive | 22.288 | 6.17 | 21.400 | 6.42 |
| WMMA + Shared Memory | 10.593 | 12.97 | 10.825 | 12.70 |
| WMMA + Shared Memory + Warp Tiling | 5.737 | 23.97 | 5.520 | 24.91 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.345 | 25.72 | 5.407 | 25.42 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.213 | 62.09 | 2.124 | 64.72 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 1.942 | 70.81 | 1.892 | 72.65 |
| WMMA Prefetch | 1.982 | 69.38 | 1.985 | 69.27 |
| rocWMMA | 2.767 | 49.69 | 28.175 | 4.88 |
| rocBLAS | 1.737 | 79.16 | 1.741 | 78.97 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 311.726 | 3.53 | 314.412 | 3.50 |
| WMMA Naive | 196.546 | 5.59 | 200.972 | 5.47 |
| WMMA + Shared Memory | 97.935 | 11.23 | 93.392 | 11.78 |
| WMMA + Shared Memory + Warp Tiling | 47.987 | 22.92 | 48.404 | 22.72 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 47.006 | 23.40 | 47.177 | 23.31 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 21.142 | 52.02 | 21.418 | 51.35 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 15.616 | 70.43 | 15.403 | 71.41 |
| WMMA Prefetch | 15.912 | 69.12 | 16.061 | 68.48 |
| rocWMMA | 23.119 | 47.58 | 227.792 | 4.83 |
| rocBLAS | 14.337 | 76.70 | 14.431 | 76.21 |

Key observations:
1. Each optimization step provides significant performance improvements
2. Global vectorized loads provide the largest single performance boost
3. Smaller matrices (1024x1024) show more variance between implementations

## Known Issues

- The WMMA HGEMM kernels using shared memory have stability issues when K > M, N (Only in Windows, in WSL2 tests pass)
- rocWMMA verification failed for 256x256 matrices in the latest tests (Only in Windows, in WSL2 tests pass)

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
