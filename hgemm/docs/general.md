# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.643 | 3.34 | 0.660 | 3.26 |
| WMMA Naive | 0.484 | 4.45 | 0.502 | 4.29 |
| WMMA + Shared Memory | 0.279 | 7.72 | 0.304 | 7.11 |
| WMMA + Shared Memory + Warp Tiling | 0.379 | 5.67 | 0.412 | 5.24 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.381 | 5.66 | 0.405 | 5.34 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.165 | 13.14 | 0.189 | 11.49 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.163 | 13.31 | 0.190 | 11.57 |
| WMMA Prefetch | 0.170 | 12.73 | 0.196 | 11.24 |
| WMMA Optimized V1 | 0.155 | 13.88 | 0.178 | 12.13 |
| WMMA Optimized V2 | 0.203 | 10.60 | 0.227 | 9.62 |
| rocBLAS | 0.099 | 21.95 | 0.126 | 17.18 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.820 | 3.57 | 4.700 | 3.66 |
| WMMA Naive | 3.760 | 4.61 | 3.640 | 4.77 |
| WMMA + Shared Memory | 1.390 | 12.34 | 1.660 | 10.40 |
| WMMA + Shared Memory + Warp Tiling | 0.839 | 20.50 | 0.846 | 20.36 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.817 | 21.04 | 0.819 | 20.99 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.410 | 42.05 | 0.426 | 40.58 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.402 | 42.81 | 0.422 | 40.92 |
| WMMA Prefetch | 0.413 | 41.65 | 0.424 | 40.74 |
| WMMA Optimized V1 | 0.380 | 45.30 | 0.406 | 42.55 |
| WMMA Optimized V2 | 0.353 | 48.87 | 0.362 | 47.62 |
| rocBLAS | 0.322 | 53.73 | 0.323 | 53.87 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.900 | 3.72 | 37.300 | 3.69 |
| WMMA Naive | 23.300 | 5.89 | 21.600 | 6.36 |
| WMMA + Shared Memory | 10.600 | 12.97 | 13.400 | 10.30 |
| WMMA + Shared Memory + Warp Tiling | 6.380 | 21.54 | 6.740 | 20.42 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.250 | 21.98 | 6.530 | 21.07 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.350 | 58.40 | 2.420 | 56.85 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.300 | 59.92 | 2.340 | 58.87 |
| WMMA Prefetch | 2.320 | 59.23 | 2.460 | 56.09 |
| WMMA Optimized V1 | 2.150 | 63.98 | 2.250 | 61.39 |
| WMMA Optimized V2 | 2.170 | 63.40 | 2.190 | 63.03 |
| rocBLAS | 1.930 | 71.37 | 1.870 | 73.82 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 324.000 | 3.40 | 327.000 | 3.37 |
| WMMA Naive | 197.000 | 5.59 | 198.000 | 5.54 |
| WMMA + Shared Memory | 94.200 | 11.68 | 94.000 | 11.70 |
| WMMA + Shared Memory + Warp Tiling | 42.900 | 25.60 | 43.000 | 25.59 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.500 | 26.52 | 41.300 | 26.60 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.700 | 62.34 | 17.500 | 62.80 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 63.92 | 17.200 | 64.08 |
| WMMA Prefetch | 17.300 | 63.74 | 17.400 | 63.19 |
| WMMA Optimized V1 | 15.800 | 69.71 | 16.100 | 68.26 |
| WMMA Optimized V2 | 14.400 | 76.63 | 14.500 | 76.12 |
| rocBLAS | 14.300 | 76.79 | 14.200 | 77.39 |

## Analysis

### Optimization Progress
My implementation shows significant performance gains through progressive optimizations:
- From the baseline shared memory implementation to WMMA Optimized V2, I achieved a **~22.5x speedup** for larger matrices
- The gap between my best implementation and rocBLAS is significantly narrow, with near-parity achieved on 8192x8192 matrices

### Platform Differences
- Windows and WSL2 performance is mostly comparable
