# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.649 | 3.32 | 0.648 | 3.32 |
| WMMA Naive | 0.488 | 4.42 | 0.532 | 4.15 |
| WMMA + Shared Memory | 0.291 | 7.47 | 0.297 | 7.38 |
| WMMA + Shared Memory + Warp Tiling | 0.416 | 5.20 | 0.401 | 5.40 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.406 | 5.35 | 0.390 | 5.55 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.177 | 12.29 | 0.175 | 12.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.170 | 12.77 | 0.201 | 11.61 |
| WMMA Prefetch | 0.176 | 12.33 | 0.226 | 10.24 |
| WMMA Optimized V1 | 0.165 | 13.15 | 0.240 | 10.46 |
| WMMA Optimized V2 | 0.211 | 10.30 | 0.303 | 8.28 |
| WMMA Optimized V3 | 0.214 | 10.12 | 0.222 | 9.79 |
| WMMA Optimized V4 | 0.211 | 10.29 | 0.204 | 10.64 |
| rocBLAS | 0.101 | 21.63 | 0.116 | 19.22 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.870 | 3.53 | 4.660 | 3.69 |
| WMMA Naive | 2.950 | 5.84 | 3.300 | 5.22 |
| WMMA + Shared Memory | 1.470 | 11.71 | 1.570 | 11.03 |
| WMMA + Shared Memory + Warp Tiling | 0.843 | 20.44 | 0.836 | 20.58 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.816 | 21.11 | 0.805 | 21.38 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.428 | 40.48 | 0.440 | 39.77 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.412 | 41.88 | 0.435 | 40.35 |
| WMMA Prefetch | 0.422 | 40.90 | 0.477 | 37.39 |
| WMMA Optimized V1 | 0.395 | 43.71 | 0.458 | 39.16 |
| WMMA Optimized V2 | 0.360 | 48.03 | 0.397 | 44.61 |
| WMMA Optimized V3 | 0.362 | 47.90 | 0.376 | 47.31 |
| WMMA Optimized V4 | 0.347 | 50.01 | 0.377 | 47.21 |
| rocBLAS | 0.309 | 55.85 | 0.329 | 53.89 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.400 | 3.78 | 41.700 | 3.35 |
| WMMA Naive | 19.100 | 7.19 | 21.600 | 6.38 |
| WMMA + Shared Memory | 11.100 | 12.40 | 12.100 | 11.32 |
| WMMA + Shared Memory + Warp Tiling | 5.720 | 24.04 | 6.530 | 21.05 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.290 | 21.85 | 6.380 | 21.55 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.360 | 58.38 | 2.450 | 56.30 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.270 | 60.47 | 2.350 | 58.67 |
| WMMA Prefetch | 2.310 | 59.61 | 2.370 | 58.02 |
| WMMA Optimized V1 | 2.160 | 63.62 | 2.250 | 61.29 |
| WMMA Optimized V2 | 2.130 | 64.52 | 2.150 | 64.15 |
| WMMA Optimized V3 | 2.140 | 64.15 | 2.150 | 63.90 |
| WMMA Optimized V4 | 2.070 | 66.32 | 2.030 | 67.69 |
| rocBLAS | 1.910 | 72.12 | 1.850 | 74.43 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 322.000 | 3.42 | 330.000 | 3.33 |
| WMMA Naive | 190.000 | 5.79 | 201.000 | 5.47 |
| WMMA + Shared Memory | 92.700 | 11.86 | 93.900 | 11.71 |
| WMMA + Shared Memory + Warp Tiling | 42.500 | 25.87 | 42.900 | 25.60 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 40.700 | 27.04 | 41.200 | 26.69 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.500 | 62.72 | 17.500 | 62.92 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.100 | 64.25 | 17.200 | 64.08 |
| WMMA Prefetch | 17.200 | 63.94 | 17.400 | 63.28 |
| WMMA Optimized V1 | 15.700 | 69.91 | 15.900 | 69.10 |
| WMMA Optimized V2 | 14.200 | 77.54 | 14.200 | 77.58 |
| WMMA Optimized V3 | 14.300 | 77.11 | 14.400 | 76.68 |
| WMMA Optimized V4 | 13.700 | 80.24 | 13.700 | 80.08 |
| rocBLAS | 14.200 | 77.37 | 14.200 | 77.44 |

## Performance for 12288x12288 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 1123.000 | 3.30 | 1123.000 | 3.30 |
| WMMA Naive | 669.000 | 5.55 | 669.000 | 5.55 |
| WMMA + Shared Memory | 313.000 | 11.87 | 313.000 | 11.87 |
| WMMA + Shared Memory + Warp Tiling | 142.000 | 26.09 | 142.000 | 26.19 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 137.000 | 27.10 | 137.000 | 27.12 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 59.700 | 62.18 | 59.700 | 62.13 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 59.100 | 62.79 | 59.100 | 62.83 |
| WMMA Prefetch | 59.900 | 61.92 | 60.100 | 61.76 |
| WMMA Optimized V1 | 53.800 | 68.95 | 54.500 | 68.14 |
| WMMA Optimized V2 | 48.000 | 77.28 | 47.700 | 77.80 |
| WMMA Optimized V3 | 47.500 | 78.06 | 47.700 | 77.82 |
| WMMA Optimized V4 | 46.200 | 80.38 | 45.900 | 80.84 |
| rocBLAS | 48.800 | 76.13 | 48.500 | 76.59 |

## Performance for 16384x16384 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 2698.000 | 3.26 | 2698.000 | 3.26 |
| WMMA Naive | 2688.000 | 3.27 | 2688.000 | 3.27 |
| WMMA + Shared Memory | 747.000 | 11.78 | 747.000 | 11.78 |
| WMMA + Shared Memory + Warp Tiling | 335.000 | 26.29 | 335.000 | 26.28 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 322.000 | 27.32 | 323.000 | 27.26 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 203.000 | 43.40 | 207.000 | 42.43 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 221.000 | 39.84 | 224.000 | 39.33 |
| WMMA Prefetch | 222.000 | 39.65 | 223.000 | 39.43 |
| WMMA Optimized V1 | 213.000 | 41.27 | 218.000 | 40.41 |
| WMMA Optimized V2 | 115.000 | 76.51 | 115.000 | 76.60 |
| WMMA Optimized V3 | 115.000 | 76.50 | 113.000 | 78.06 |
| WMMA Optimized V4 | 110.000 | 80.31 | 110.000 | 80.31 |
| rocBLAS | 203.000 | 43.32 | 205.000 | 42.92 |

## Analysis

### Optimization Progress
- From the baseline shared memory implementation to the best optimized version, achieved a **~23x speedup** for larger matrices
- WMMA Optimized V4 is the best performing implementation across all tested matrix sizes above 1024x1024
  - 1024x1024 requires tuning to smaller tiles

### Platform Differences
- Windows and WSL2 performance is mostly comparable
