# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.631 | 3.41 | 0.643 | 3.35 |
| WMMA Naive | 0.483 | 4.45 | 0.471 | 4.57 |
| WMMA + Shared Memory | 0.273 | 7.88 | 0.326 | 6.60 |
| WMMA + Shared Memory + Warp Tiling | 0.373 | 5.77 | 0.387 | 5.56 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.371 | 5.80 | 0.380 | 5.67 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.162 | 13.28 | 0.175 | 12.29 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.164 | 13.11 | 0.170 | 12.65 |
| WMMA Prefetch | 0.168 | 12.81 | 0.178 | 12.08 |
| WMMA Optimized V1 | 0.154 | 13.96 | 0.164 | 13.11 |
| WMMA Optimized V2 | 0.202 | 10.65 | 0.212 | 10.14 |
| rocBLAS | 0.097 | 22.14 | 0.113 | 19.03 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 5.040 | 3.40 | 4.740 | 3.62 |
| WMMA Naive | 3.210 | 5.34 | 3.490 | 4.91 |
| WMMA + Shared Memory | 1.360 | 12.61 | 1.670 | 10.27 |
| WMMA + Shared Memory + Warp Tiling | 0.844 | 20.31 | 0.801 | 21.39 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.801 | 21.39 | 0.774 | 22.14 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.403 | 42.52 | 0.403 | 42.52 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.394 | 43.50 | 0.395 | 43.39 |
| WMMA Prefetch | 0.411 | 41.70 | 0.397 | 43.17 |
| WMMA Optimized V1 | 0.375 | 45.71 | 0.373 | 45.96 |
| WMMA Optimized V2 | 0.348 | 49.28 | 0.353 | 48.57 |
| rocBLAS | 0.316 | 54.25 | 0.307 | 55.81 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.600 | 3.76 | 41.600 | 3.30 |
| WMMA Naive | 21.800 | 6.30 | 22.200 | 6.19 |
| WMMA + Shared Memory | 10.400 | 13.22 | 12.100 | 11.36 |
| WMMA + Shared Memory + Warp Tiling | 6.300 | 21.82 | 6.690 | 20.55 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.190 | 22.21 | 6.440 | 21.34 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.350 | 58.47 | 2.500 | 54.96 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.300 | 59.73 | 2.380 | 57.75 |
| WMMA Prefetch | 2.310 | 59.47 | 2.380 | 57.75 |
| WMMA Optimized V1 | 2.150 | 63.93 | 2.170 | 63.34 |
| WMMA Optimized V2 | 2.180 | 63.05 | 2.210 | 62.15 |
| rocBLAS | 1.920 | 71.58 | 1.850 | 74.29 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 324.000 | 3.39 | 327.000 | 3.36 |
| WMMA Naive | 196.000 | 5.61 | 200.000 | 5.50 |
| WMMA + Shared Memory | 93.700 | 11.74 | 94.200 | 11.67 |
| WMMA + Shared Memory + Warp Tiling | 42.300 | 25.99 | 42.700 | 25.75 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 40.900 | 26.89 | 40.800 | 26.96 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.600 | 62.47 | 17.600 | 62.47 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.200 | 63.93 | 17.300 | 63.56 |
| WMMA Prefetch | 17.200 | 63.93 | 17.400 | 63.19 |
| WMMA Optimized V1 | 15.700 | 70.03 | 16.000 | 68.72 |
| WMMA Optimized V2 | 14.700 | 74.80 | 14.800 | 74.29 |
| rocBLAS | 14.300 | 76.89 | 14.200 | 77.42 |

## Analysis

### Optimization Progress
My implementation shows significant performance gains through progressive optimizations:
- From the baseline shared memory implementation to WMMA Optimized V2, I achieved a **~20x speedup** for larger matrices
- The gap between my best implementation and rocBLAS narrows as matrix size increases

### Platform Differences
- Windows and WSL2 performance is mostly comparable
- At larger sizes (8192x8192), the difference between platforms becomes negligible
