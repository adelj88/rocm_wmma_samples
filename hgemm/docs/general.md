# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows (HIP SDK 6.2.4) and WSL2 (Ubuntu 24.04.1 LTS, ROCm 6.4.1). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.614 | 3.50 | 0.660 | 3.48 |
| WMMA Naive | 0.505 | 4.31 | 0.454 | 4.75 |
| WMMA + Shared Memory | 0.268 | 8.03 | 0.277 | 8.04 |
| WMMA + Shared Memory + Warp Tiling | 0.364 | 5.92 | 0.370 | 5.84 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.428 | 5.11 | 0.365 | 5.91 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.181 | 12.17 | 0.168 | 12.95 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.159 | 13.56 | 0.172 | 13.08 |
| WMMA Prefetch | 0.170 | 12.71 | 0.169 | 12.82 |
| WMMA Optimized V1 | 0.153 | 14.12 | 0.158 | 13.77 |
| WMMA Optimized V2 | 0.194 | 11.08 | 0.203 | 10.72 |
| WMMA Optimized V3 | 0.199 | 10.82 | 0.209 | 10.41 |
| WMMA Optimized V4 | 0.191 | 11.30 | 0.195 | 11.10 |
| WMMA Optimized V5 | 0.144 | 14.97 | 0.146 | 14.85 |
| rocBLAS | 0.097 | 22.45 | 0.100 | 21.75 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.68 | 3.68 | 4.61 | 3.73 |
| WMMA Naive | 3.20 | 5.39 | 3.23 | 5.33 |
| WMMA + Shared Memory | 1.64 | 10.75 | 1.50 | 11.49 |
| WMMA + Shared Memory + Warp Tiling | 0.798 | 21.54 | 0.784 | 21.94 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.793 | 21.72 | 0.758 | 22.70 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.418 | 41.34 | 0.387 | 44.57 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.384 | 44.95 | 0.379 | 45.47 |
| WMMA Prefetch | 0.399 | 43.20 | 0.386 | 44.65 |
| WMMA Optimized V1 | 0.365 | 47.17 | 0.362 | 47.61 |
| WMMA Optimized V2 | 0.335 | 51.38 | 0.329 | 52.51 |
| WMMA Optimized V3 | 0.339 | 50.76 | 0.328 | 52.52 |
| WMMA Optimized V4 | 0.326 | 52.78 | 0.322 | 53.61 |
| WMMA Optimized V5 | 0.330 | 52.18 | 0.325 | 53.06 |
| rocBLAS | 0.308 | 55.96 | 0.286 | 60.23 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 36.2 | 3.80 | 44.0 | 3.14 |
| WMMA Naive | 20.0 | 6.89 | 20.4 | 6.76 |
| WMMA + Shared Memory | 11.4 | 12.06 | 11.8 | 11.65 |
| WMMA + Shared Memory + Warp Tiling | 6.64 | 20.73 | 6.41 | 21.82 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 5.57 | 24.68 | 6.13 | 22.44 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.35 | 58.63 | 2.34 | 58.81 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.28 | 60.33 | 2.24 | 61.31 |
| WMMA Prefetch | 2.49 | 55.65 | 2.26 | 60.77 |
| WMMA Optimized V1 | 2.15 | 64.01 | 2.12 | 64.94 |
| WMMA Optimized V2 | 2.46 | 56.94 | 2.10 | 65.51 |
| WMMA Optimized V3 | 2.52 | 55.51 | 2.13 | 64.73 |
| WMMA Optimized V4 | 2.38 | 58.81 | 2.03 | 67.75 |
| WMMA Optimized V5 | 2.00 | 69.49 | 1.86 | 74.12 |
| rocBLAS | 2.09 | 66.75 | 1.86 | 74.03 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 310.0 | 3.55 | 313.0 | 3.52 |
| WMMA Naive | 190.0 | 5.77 | 192.0 | 5.73 |
| WMMA + Shared Memory | 93.6 | 11.74 | 93.5 | 11.75 |
| WMMA + Shared Memory + Warp Tiling | 42.2 | 26.03 | 42.1 | 26.09 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 40.7 | 27.03 | 40.2 | 27.37 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.8 | 61.77 | 17.8 | 62.01 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.7 | 62.15 | 17.3 | 63.76 |
| WMMA Prefetch | 17.5 | 62.81 | 17.5 | 62.98 |
| WMMA Optimized V1 | 16.1 | 68.44 | 16.1 | 68.20 |
| WMMA Optimized V2 | 14.8 | 74.16 | 14.4 | 76.26 |
| WMMA Optimized V3 | 14.9 | 73.75 | 14.6 | 75.62 |
| WMMA Optimized V4 | 14.4 | 76.37 | 14.0 | 78.54 |
| WMMA Optimized V5 | 14.4 | 76.36 | 14.2 | 77.79 |
| rocBLAS | 14.7 | 75.13 | 14.3 | 76.98 |

## Analysis

### Optimization Progress
- From the baseline shared memory implementation to the best optimized version, achieved a **~22x speedup** for larger matrices
- WMMA Optimized V5 shows the best performance for 4096x4096 matrices
- WMMA Optimized V4 performs best for 8192x8192 matrices
- Performance is now very close to rocBLAS across all matrix sizes

### Platform Differences
- Windows and WSL2 performance is mostly comparable
- For larger matrices (4096x4096 and above), WSL2 sometimes shows slightly better performance
