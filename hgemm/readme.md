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
| Shared Memory | 0.558 | 3.87 | 0.561 | 3.85 |
| WMMA Naive | 0.399 | 5.42 | 0.408 | 5.30 |
| WMMA + Shared Memory | 0.185 | 11.68 | 0.205 | 10.56 |
| WMMA + Shared Memory + Warp Tiling | 0.281 | 7.69 | 0.362 | 5.98 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.203 | 10.63 | 0.250 | 8.64 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.079 | 27.36 | 0.117 | 18.50 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.062 | 35.00 | 0.078 | 27.57 |
| WMMA Prefetch | 0.062 | 34.94 | 0.076 | 28.44 |
| WMMA Decoupled | 0.057 | 38.15 | 0.070 | 30.87 |
| rocWMMA | 0.081 | 26.60 | 0.261 | 8.29 |
| rocBLAS | 0.054 | 40.05 | 0.043 | 49.78 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.383 | 3.93 | 4.559 | 3.78 |
| WMMA Naive | 3.188 | 5.40 | 3.015 | 5.71 |
| WMMA + Shared Memory | 1.334 | 12.90 | 1.353 | 12.71 |
| WMMA + Shared Memory + Warp Tiling | 0.937 | 18.38 | 0.966 | 17.83 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.918 | 18.75 | 0.962 | 17.90 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.370 | 46.48 | 0.370 | 46.52 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.336 | 51.19 | 0.340 | 50.58 |
| WMMA Prefetch | 0.330 | 52.12 | 0.346 | 49.70 |
| WMMA Decoupled | 0.315 | 54.69 | 0.323 | 53.39 |
| rocWMMA | 0.416 | 41.41 | 3.548 | 4.85 |
| rocBLAS | 0.252 | 68.22 | 0.236 | 72.85 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 35.844 | 3.83 | 36.389 | 3.78 |
| WMMA Naive | 21.856 | 6.29 | 21.269 | 6.46 |
| WMMA + Shared Memory | 11.019 | 12.47 | 10.811 | 12.72 |
| WMMA + Shared Memory + Warp Tiling | 6.845 | 20.08 | 6.829 | 20.13 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.532 | 24.84 | 5.317 | 25.83 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.109 | 65.19 | 2.147 | 64.04 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 1.894 | 72.56 | 1.873 | 73.34 |
| WMMA Prefetch | 1.967 | 69.88 | 1.959 | 70.11 |
| WMMA Decoupled | 1.868 | 73.57 | 1.853 | 74.14 |
| rocWMMA | 2.757 | 49.87 | 28.163 | 4.88 |
| rocBLAS | 1.737 | 79.15 | 1.741 | 78.94 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 310.772 | 3.54 | 313.708 | 3.51 |
| WMMA Naive | 196.323 | 5.60 | 199.719 | 5.51 |
| WMMA + Shared Memory | 96.578 | 11.39 | 97.271 | 11.31 |
| WMMA + Shared Memory + Warp Tiling | 47.793 | 23.01 | 48.325 | 22.76 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 46.616 | 23.59 | 46.947 | 23.42 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 21.125 | 52.06 | 21.205 | 51.87 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 15.548 | 70.73 | 15.389 | 71.47 |
| WMMA Prefetch | 15.851 | 69.38 | 16.011 | 68.69 |
| WMMA Decoupled | 15.189 | 72.41 | 15.104 | 72.82 |
| rocWMMA | 23.019 | 47.78 | 227.666 | 4.83 |
| rocBLAS | 14.267 | 77.09 | 14.399 | 76.38 |

Key observations:
1. Each optimization step provides significant performance improvements
2. Global vectorized loads provide the largest single performance boost
3. Smaller matrices (1024x1024) show more variance between implementations
4. rocWMMA performs well on Windows but shows significantly reduced performance on WSL2

## Known Issues

- The WMMA HGEMM kernels using shared memory have stability issues when K > M, N (Only in Windows, in WSL2 tests pass)
- WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads verification failed for 256x256 matrices in WSL2

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
