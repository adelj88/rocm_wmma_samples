# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.664 | 3.24 | 0.669 | 3.25 |
| WMMA Naive | 0.505 | 4.27 | 0.505 | 4.28 |
| WMMA + Shared Memory | 0.312 | 6.96 | 0.319 | 7.04 |
| WMMA + Shared Memory + Warp Tiling | 0.421 | 5.16 | 0.412 | 5.24 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.425 | 5.12 | 0.413 | 5.29 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.192 | 11.27 | 0.202 | 11.19 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.184 | 11.74 | 0.194 | 11.29 |
| WMMA Prefetch | 0.200 | 10.88 | 0.192 | 11.24 |
| WMMA Optimized V1 | 0.183 | 11.90 | 0.187 | 11.89 |
| WMMA Optimized V2 | 0.225 | 9.59 | 0.229 | 9.50 |
| rocBLAS | 0.129 | 16.96 | 0.134 | 16.88 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.650 | 3.69 | 4.610 | 3.73 |
| WMMA Naive | 3.290 | 5.23 | 3.440 | 5.01 |
| WMMA + Shared Memory | 1.500 | 11.46 | 1.690 | 10.20 |
| WMMA + Shared Memory + Warp Tiling | 0.883 | 19.51 | 0.845 | 20.36 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.862 | 20.00 | 0.818 | 21.04 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.431 | 40.04 | 0.422 | 40.96 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.419 | 41.07 | 0.422 | 41.15 |
| WMMA Prefetch | 0.444 | 38.85 | 0.443 | 39.56 |
| WMMA Optimized V1 | 0.413 | 41.92 | 0.410 | 42.34 |
| WMMA Optimized V2 | 0.387 | 44.93 | 0.377 | 46.36 |
| rocBLAS | 0.342 | 50.86 | 0.324 | 53.44 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.500 | 3.76 | 36.900 | 3.73 |
| WMMA Naive | 22.200 | 6.20 | 21.300 | 6.45 |
| WMMA + Shared Memory | 11.300 | 12.19 | 13.300 | 10.34 |
| WMMA + Shared Memory + Warp Tiling | 6.450 | 21.31 | 6.390 | 21.50 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.320 | 21.76 | 6.380 | 21.55 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.390 | 57.66 | 2.450 | 56.32 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.330 | 58.94 | 2.380 | 57.94 |
| WMMA Prefetch | 2.350 | 58.61 | 2.440 | 56.51 |
| WMMA Optimized V1 | 2.170 | 63.28 | 2.240 | 61.55 |
| WMMA Optimized V2 | 2.240 | 61.48 | 2.280 | 60.46 |
| rocBLAS | 1.940 | 70.88 | 1.860 | 74.23 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 324.000 | 3.39 | 326.000 | 3.37 |
| WMMA Naive | 195.000 | 5.64 | 199.000 | 5.53 |
| WMMA + Shared Memory | 93.900 | 11.71 | 94.100 | 11.68 |
| WMMA + Shared Memory + Warp Tiling | 42.800 | 25.70 | 42.700 | 25.76 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.400 | 26.57 | 41.200 | 26.67 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.600 | 62.54 | 17.500 | 62.72 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.300 | 63.69 | 17.200 | 64.17 |
| WMMA Prefetch | 17.300 | 63.66 | 17.500 | 62.97 |
| WMMA Optimized V1 | 15.800 | 69.84 | 16.000 | 68.70 |
| WMMA Optimized V2 | 14.800 | 74.54 | 14.900 | 73.86 |
| rocBLAS | 14.300 | 76.74 | 14.200 | 77.46 |

## Analysis

### Optimization Progress
My implementation shows significant performance gains through progressive optimizations:
- From the baseline shared memory implementation to WMMA Optimized V2, I achieved a **~20x speedup** for larger matrices
- The gap between my best implementation and rocBLAS narrows as matrix size increases

### Platform Differences
- Windows and WSL2 performance is mostly comparable
- At larger sizes (8192x8192), the difference between platforms becomes negligible