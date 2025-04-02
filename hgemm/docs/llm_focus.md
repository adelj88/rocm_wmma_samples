# LLM-Focused Benchmarks

The best HGEMM implementation (`wmma_opt_2`) has been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

Below are benchmarks comparing my best implementation against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_2` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` Ratio |
|------------------|----------------|-----------------|-------------------|-------------------|
| m=4096, n=4096, k=1024 | QKV Projection | 50.96 | 64.34 | 79.2% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 59.69 | 72.50 | 82.3% |
| m=4096, n=2048, k=64 | Attention Score | 9.30 | 9.22 | 100.8% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 31.99 | 37.42 | 85.5% |
| m=4096, n=16384, k=4096 | FFN First Layer | 73.29 | 75.38 | 97.2% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 64.40 | 53.85 | 119.6% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 67.22 | 63.39 | 106.0% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 73.69 | 73.23 | 100.6% |
| m=32768, n=4096, k=4096 | Long Context Processing | 71.59 | 76.18 | 94.0% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 72.02 | 61.91 | 116.3% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:1024}/manual_time       0.675 ms        0.671 ms         1001 TFLOPS=50.9622 bytes_per_second=69.4426Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:8192,k:1024}/manual_time        2.35 ms         2.33 ms          356 TFLOPS=59.6854 bytes_per_second=66.621Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:2048,k:64}/manual_time         0.116 ms        0.117 ms         6033 TFLOPS=9.29552 bytes_per_second=140.419Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:4096,k:128}/manual_time        0.269 ms        0.270 ms         2603 TFLOPS=31.9881 bytes_per_second=242.976Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:16384,k:4096}/manual_time       7.50 ms         7.57 ms           95 TFLOPS=73.2916 bytes_per_second=37.4814Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:16384}/manual_time       8.54 ms         8.37 ms           84 TFLOPS=64.3991 bytes_per_second=32.9259Gi/s
{hgemm:kernel_type::wmma_opt_2,m:2048,n:5120,k:5120}/manual_time        1.60 ms         1.60 ms          439 TFLOPS=67.2162 bytes_per_second=54.9994Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:5120,k:5120}/manual_time        2.92 ms         2.91 ms          247 TFLOPS=73.6857 bytes_per_second=43.5175Gi/s
{hgemm:kernel_type::wmma_opt_2,m:32768,n:4096,k:4096}/manual_time       15.4 ms         15.3 ms           49 TFLOPS=71.5899 bytes_per_second=34.5633Gi/s
{hgemm:kernel_type::wmma_opt_2,m:65536,n:2048,k:2048}/manual_time       7.63 ms         7.64 ms           92 TFLOPS=72.0239 bytes_per_second=66.5216Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.537 ms        0.513 ms         1371 TFLOPS=64.3416 bytes_per_second=87.2735Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.90 ms         1.91 ms          336 TFLOPS=72.4994 bytes_per_second=82.3112Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.118 ms        0.121 ms         6068 TFLOPS=9.22284 bytes_per_second=138.191Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.234 ms        0.238 ms         3088 TFLOPS=37.419 bytes_per_second=280.199Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.30 ms         7.43 ms          101 TFLOPS=75.3799 bytes_per_second=38.5154Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.2 ms         10.3 ms           67 TFLOPS=53.8462 bytes_per_second=27.4908Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.74 ms         1.70 ms          487 TFLOPS=63.3916 bytes_per_second=50.6015Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.94 ms         2.89 ms          227 TFLOPS=73.2263 bytes_per_second=43.2324Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.4 ms         14.4 ms           51 TFLOPS=76.1827 bytes_per_second=36.7672Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.88 ms         8.90 ms           79 TFLOPS=61.9056 bytes_per_second=57.1776Gi/s
```
