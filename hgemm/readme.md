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
| Shared Memory | 0.559 | 3.86 |
| WMMA Naive | 0.402 | 5.37 |
| WMMA + Shared Memory | 0.184 | 11.73 |
| WMMA + Shared Memory + Warp Tiling | 0.253 | 8.54 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.204 | 10.60 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.082 | 26.38 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.068 | 31.63 |
| WMMA Prefetch | 0.063 | 34.38 |
| rocWMMA | 0.076 | 28.44 |
| rocBLAS | 0.057 | 37.87 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 4.539 | 3.79 |
| WMMA Naive | 3.215 | 5.35 |
| WMMA + Shared Memory | 1.326 | 12.96 |
| WMMA + Shared Memory + Warp Tiling | 0.938 | 18.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.919 | 18.71 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.364 | 47.29 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.346 | 49.71 |
| WMMA Prefetch | 0.330 | 52.05 |
| rocWMMA | 0.410 | 41.93 |
| rocBLAS | 0.252 | 68.22 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 35.775 | 3.84 |
| WMMA Naive | 21.995 | 6.25 |
| WMMA + Shared Memory | 12.861 | 10.69 |
| WMMA + Shared Memory + Warp Tiling | 5.375 | 25.56 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.272 | 26.06 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.212 | 62.11 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 1.972 | 69.61 |
| WMMA Prefetch | 1.962 | 70.03 |
| rocWMMA | 2.847 | 48.24 |
| rocBLAS | 1.752 | 78.43 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Time (ms) | TFLOPs/s |
|----------------|-----------|----------|
| Shared Memory | 309.566 | 3.55 |
| WMMA Naive | 195.489 | 5.63 |
| WMMA + Shared Memory | 96.539 | 11.39 |
| WMMA + Shared Memory + Warp Tiling | 47.833 | 23.00 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 46.646 | 23.58 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 21.134 | 52.04 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 15.982 | 68.81 |
| WMMA Prefetch | 15.816 | 69.54 |
| rocWMMA | 22.983 | 47.85 |
| rocBLAS | 14.283 | 77.00 |

Key observations:
1. Each optimization step provides significant performance improvements
2. Global vectorized loads provide the largest single performance boost
3. Smaller matrices (1024x1024) show more variance between implementations

## Known Issues

- The WMMA HGEMM kernels using shared memory have stability issues when K > M, N
- rocWMMA verification failed for 256x256 matrices in the latest tests

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

2. **Bug Fixes:**
   - Address WMMA + shared memory issues when K > M, N
   - Investigate rocWMMA verification failure on 256x256 matrices
