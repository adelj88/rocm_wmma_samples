# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.630 | 3.41 | 0.684 | 3.20 |
| WMMA Naive | 0.485 | 4.46 | 0.466 | 4.63 |
| WMMA + Shared Memory | 0.292 | 7.45 | 0.313 | 7.66 |
| WMMA + Shared Memory + Warp Tiling | 0.401 | 5.41 | 0.398 | 5.47 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.400 | 5.42 | 0.419 | 5.45 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.174 | 12.47 | 0.191 | 11.80 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.174 | 12.55 | 0.170 | 12.76 |
| WMMA Prefetch | 0.171 | 12.67 | 0.181 | 12.22 |
| WMMA Optimized V1 | 0.163 | 13.31 | 0.165 | 13.32 |
| WMMA Optimized V2 | 0.208 | 10.42 | 0.216 | 10.55 |
| WMMA Optimized V3 | 0.212 | 10.22 | 0.223 | 10.07 |
| rocBLAS | 0.133 | 17.21 | 0.115 | 19.58 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.710 | 3.65 | 4.610 | 3.73 |
| WMMA Naive | 3.250 | 5.30 | 3.660 | 4.74 |
| WMMA + Shared Memory | 1.510 | 11.40 | 1.660 | 10.38 |
| WMMA + Shared Memory + Warp Tiling | 0.847 | 20.34 | 0.805 | 21.37 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.809 | 21.28 | 0.802 | 21.76 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.422 | 40.98 | 0.406 | 42.65 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.415 | 41.67 | 0.400 | 43.16 |
| WMMA Prefetch | 0.422 | 40.99 | 0.427 | 40.75 |
| WMMA Optimized V1 | 0.391 | 44.30 | 0.387 | 44.91 |
| WMMA Optimized V2 | 0.362 | 47.88 | 0.352 | 49.85 |
| WMMA Optimized V3 | 0.365 | 47.56 | 0.354 | 49.83 |
| rocBLAS | 0.339 | 51.39 | 0.300 | 57.79 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.600 | 3.76 | 42.900 | 3.25 |
| WMMA Naive | 21.900 | 6.28 | 21.700 | 6.35 |
| WMMA + Shared Memory | 10.500 | 13.14 | 12.300 | 11.23 |
| WMMA + Shared Memory + Warp Tiling | 5.810 | 23.68 | 6.450 | 21.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.440 | 21.34 | 6.370 | 21.59 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.390 | 57.66 | 2.310 | 59.59 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.310 | 59.66 | 2.470 | 56.03 |
| WMMA Prefetch | 2.330 | 59.06 | 2.390 | 57.77 |
| WMMA Optimized V1 | 2.150 | 63.89 | 2.180 | 63.31 |
| WMMA Optimized V2 | 2.180 | 63.09 | 2.450 | 60.19 |
| WMMA Optimized V3 | 2.160 | 63.54 | 2.330 | 59.43 |
| rocBLAS | 1.940 | 70.94 | 1.840 | 74.86 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 324.000 | 3.39 | 325.000 | 3.39 |
| WMMA Naive | 197.000 | 5.57 | 198.000 | 5.55 |
| WMMA + Shared Memory | 94.200 | 11.68 | 94.100 | 11.68 |
| WMMA + Shared Memory + Warp Tiling | 42.500 | 25.85 | 42.500 | 25.88 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.100 | 26.74 | 40.800 | 26.94 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.600 | 62.55 | 17.500 | 62.87 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 64.12 | 17.200 | 64.08 |
| WMMA Prefetch | 17.200 | 63.83 | 17.500 | 62.91 |
| WMMA Optimized V1 | 15.800 | 69.90 | 16.000 | 68.91 |
| WMMA Optimized V2 | 14.300 | 76.91 | 14.300 | 76.73 |
| WMMA Optimized V3 | 14.300 | 76.90 | 14.300 | 77.12 |
| rocBLAS | 14.300 | 76.99 | 14.200 | 77.39 |

## Performance for 12288x12288 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 1102.000 | 3.37 | 1105.000 | 3.36 |
| WMMA Naive | 688.000 | 5.39 | 718.000 | 5.17 |
| WMMA + Shared Memory | 316.000 | 11.73 | 318.000 | 11.68 |
| WMMA + Shared Memory + Warp Tiling | 142.000 | 26.15 | 142.000 | 26.13 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 137.000 | 27.07 | 136.000 | 27.20 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 60.000 | 61.81 | 59.900 | 61.94 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 59.400 | 62.51 | 59.300 | 62.58 |
| WMMA Prefetch | 60.200 | 61.69 | 60.500 | 61.38 |
| WMMA Optimized V1 | 54.000 | 68.68 | 54.800 | 67.70 |
| WMMA Optimized V2 | 49.700 | 74.66 | 49.800 | 74.55 |
| WMMA Optimized V3 | 48.000 | 77.24 | 47.900 | 77.40 |
| rocBLAS | 48.800 | 76.05 | 48.600 | 76.39 |

## Performance for 16384x16384 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 2629.000 | 3.35 | 2641.000 | 3.33 |
| WMMA Naive | 2672.000 | 3.29 | 2664.000 | 3.30 |
| WMMA + Shared Memory | 756.000 | 11.63 | 749.000 | 11.74 |
| WMMA + Shared Memory + Warp Tiling | 335.000 | 26.23 | 334.000 | 26.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 328.000 | 26.80 | 323.000 | 27.27 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 203.000 | 43.45 | 203.000 | 43.36 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 219.000 | 40.09 | 220.000 | 39.97 |
| WMMA Prefetch | 223.000 | 39.51 | 221.000 | 39.88 |
| WMMA Optimized V1 | 217.000 | 40.53 | 212.000 | 41.59 |
| WMMA Optimized V2 | 115.000 | 76.22 | 116.000 | 75.87 |
| WMMA Optimized V3 | 115.000 | 76.21 | 116.000 | 76.09 |
| rocBLAS | 202.000 | 43.53 | 203.000 | 43.32 |

## Analysis

### Optimization Progress
- From the baseline shared memory implementation to the best optimized version, achieved a **~22.5x speedup** for larger matrices
- WMMA Optimized V2 and V3 are now the best performing implementations for large matrices (12288x12288 and 16384x16384)
- The gap between the best implementation and rocBLAS has closed significantly

### Platform Differences
- Windows and WSL2 performance is mostly comparable with minor variations
