# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.626 | 3.43 | 0.643 | 3.34 |
| WMMA Naive | 0.467 | 4.61 | 0.467 | 4.61 |
| WMMA + Shared Memory | 0.276 | 7.83 | 0.281 | 7.68 |
| WMMA + Shared Memory + Warp Tiling | 0.376 | 5.75 | 0.386 | 5.59 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.372 | 5.81 | 0.382 | 5.65 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.163 | 13.32 | 0.175 | 12.44 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.164 | 13.27 | 0.173 | 12.71 |
| WMMA Prefetch | 0.171 | 12.70 | 0.208 | 11.30 |
| WMMA Optimized V1 | 0.156 | 13.93 | 0.173 | 13.03 |
| WMMA Optimized V2 | 0.202 | 10.74 | 0.204 | 10.59 |
| WMMA Optimized V3 | 0.204 | 10.63 | 0.220 | 10.42 |
| rocBLAS | 0.099 | 22.07 | 0.111 | 20.22 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.650 | 3.69 | 4.830 | 3.57 |
| WMMA Naive | 3.480 | 4.94 | 3.480 | 4.97 |
| WMMA + Shared Memory | 1.460 | 12.00 | 1.620 | 10.66 |
| WMMA + Shared Memory + Warp Tiling | 0.834 | 20.63 | 0.817 | 21.07 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.803 | 21.41 | 0.789 | 21.79 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.403 | 42.74 | 0.403 | 42.78 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.395 | 43.69 | 0.391 | 44.10 |
| WMMA Prefetch | 0.410 | 42.11 | 0.442 | 40.89 |
| WMMA Optimized V1 | 0.377 | 45.77 | 0.397 | 44.88 |
| WMMA Optimized V2 | 0.348 | 49.64 | 0.339 | 50.99 |
| WMMA Optimized V3 | 0.347 | 49.81 | 0.341 | 50.56 |
| rocBLAS | 0.320 | 54.18 | 0.302 | 58.20 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.900 | 3.72 | 41.000 | 3.40 |
| WMMA Naive | 23.300 | 5.91 | 22.000 | 6.25 |
| WMMA + Shared Memory | 10.400 | 13.21 | 12.400 | 11.13 |
| WMMA + Shared Memory + Warp Tiling | 6.310 | 21.77 | 6.780 | 20.65 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.200 | 22.16 | 6.360 | 21.64 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.360 | 58.20 | 2.450 | 56.32 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.300 | 59.84 | 2.310 | 59.69 |
| WMMA Prefetch | 2.350 | 58.52 | 2.350 | 58.57 |
| WMMA Optimized V1 | 2.160 | 63.62 | 2.160 | 63.83 |
| WMMA Optimized V2 | 2.150 | 64.09 | 2.100 | 65.47 |
| WMMA Optimized V3 | 2.160 | 63.82 | 2.180 | 63.26 |
| rocBLAS | 1.940 | 70.87 | 1.860 | 74.32 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 325.000 | 3.39 | 329.000 | 3.34 |
| WMMA Naive | 196.000 | 5.61 | 200.000 | 5.48 |
| WMMA + Shared Memory | 94.000 | 11.70 | 94.100 | 11.68 |
| WMMA + Shared Memory + Warp Tiling | 42.800 | 25.72 | 42.800 | 25.71 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 40.900 | 26.87 | 41.000 | 26.84 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.600 | 62.48 | 17.500 | 62.97 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 64.04 | 17.100 | 64.24 |
| WMMA Prefetch | 17.600 | 62.49 | 17.400 | 63.33 |
| WMMA Optimized V1 | 15.700 | 70.01 | 16.100 | 68.45 |
| WMMA Optimized V2 | 14.200 | 77.37 | 14.400 | 76.70 |
| WMMA Optimized V3 | 14.300 | 77.06 | 14.400 | 76.25 |
| rocBLAS | 14.300 | 76.73 | 14.200 | 77.54 |

## Performance for 12288x12288 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 1098.000 | 3.38 | 1114.000 | 3.33 |
| WMMA Naive | 699.000 | 5.31 | 714.000 | 5.20 |
| WMMA + Shared Memory | 316.000 | 11.74 | 318.000 | 11.67 |
| WMMA + Shared Memory + Warp Tiling | 142.000 | 26.07 | 142.000 | 26.07 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 137.000 | 27.11 | 137.000 | 27.06 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 59.800 | 62.07 | 59.800 | 62.07 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 59.500 | 62.41 | 59.100 | 62.84 |
| WMMA Prefetch | 59.900 | 61.91 | 60.500 | 61.38 |
| WMMA Optimized V1 | 54.100 | 68.55 | 54.500 | 68.09 |
| WMMA Optimized V2 | 48.100 | 77.13 | 48.000 | 77.29 |
| WMMA Optimized V3 | 47.700 | 77.86 | 48.000 | 77.39 |
| rocBLAS | 48.900 | 75.92 | 48.800 | 76.13 |

## Performance for 16384x16384 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 2666.000 | 3.30 | 2686.000 | 3.27 |
| WMMA Naive | 2671.000 | 3.29 | 2667.000 | 3.30 |
| WMMA + Shared Memory | 760.000 | 11.58 | 749.000 | 11.74 |
| WMMA + Shared Memory + Warp Tiling | 338.000 | 26.06 | 335.000 | 26.29 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 327.000 | 26.90 | 324.000 | 27.13 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 205.000 | 42.98 | 206.000 | 42.69 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 223.000 | 39.49 | 222.000 | 39.67 |
| WMMA Prefetch | 221.000 | 39.83 | 223.000 | 39.38 |
| WMMA Optimized V1 | 216.000 | 40.65 | 215.000 | 40.97 |
| WMMA Optimized V2 | 116.000 | 75.92 | 115.000 | 76.45 |
| WMMA Optimized V3 | 116.000 | 76.11 | 115.000 | 76.22 |
| rocBLAS | 203.000 | 43.40 | 204.000 | 43.17 |

## Analysis

### Optimization Progress
- From the baseline shared memory implementation to the best optimized version, achieved a **~23x speedup** for larger matrices
- WMMA Optimized V2 and V3 are now the best performing implementations across all tested matrix sizes above 1024x1024
  - 1024x1024 requires tuning to smaller tiles

### Platform Differences
- Windows and WSL2 performance is mostly comparable
- For the largest matrix size (16384x16384), the optimized implementations show almost identical performance across both platforms
