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

Performance measured on AMD Radeon RX 7900 GRE on Windows (HIP SDK 6.2.4). All implementations use half precision (FP16).

### Performance for 1024x1024 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 0.575 | 3.75 |
| WMMA Naive | 0.408 | 5.30 |
| WMMA + Shared Memory | 0.186 | 11.61 |
| WMMA + Shared Memory + Warp Tiling | 0.261 | 8.28 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.207 | 10.45 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.079 | 27.32 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.064 | 33.90 |
| WMMA Prefetch | 0.063 | 34.41 |
| rocWMMA | 0.077 | 28.07 |
| rocBLAS | 0.046 | 46.91 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 4.670 | 3.68 |
| WMMA Naive | 3.277 | 5.25 |
| WMMA + Shared Memory | 1.357 | 12.68 |
| WMMA + Shared Memory + Warp Tiling | 0.960 | 17.91 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.943 | 18.23 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.391 | 44.00 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.345 | 49.85 |
| WMMA Prefetch | 0.337 | 51.06 |
| rocWMMA | 0.416 | 41.36 |
| rocBLAS | 0.254 | 67.81 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 36.408 | 3.78 |
| WMMA Naive | 22.288 | 6.17 |
| WMMA + Shared Memory | 10.593 | 12.97 |
| WMMA + Shared Memory + Warp Tiling | 5.737 | 23.97 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.345 | 25.72 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.213 | 62.09 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 1.942 | 70.81 |
| WMMA Prefetch | 1.982 | 69.38 |
| rocWMMA | 2.767 | 49.69 |
| rocBLAS | 1.737 | 79.16 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 311.726 | 3.53 |
| WMMA Naive | 196.546 | 5.59 |
| WMMA + Shared Memory | 97.935 | 11.23 |
| WMMA + Shared Memory + Warp Tiling | 47.987 | 22.92 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 47.006 | 23.40 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 21.142 | 52.02 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 15.616 | 70.43 |
| WMMA Prefetch | 15.912 | 69.12 |
| rocWMMA | 23.119 | 47.58 |
| rocBLAS | 14.337 | 76.70 |

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
