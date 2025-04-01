# LLM-Focused Benchmarks

The best HGEMM implementation (`wmma_opt_2`) has been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

Below are benchmarks comparing my best implementation against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_2` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` Ratio |
|------------------|----------------|-----------------|-------------------|-------------------|
| m=4096, n=4096, k=1024 | QKV Projection | 52.76 | 64.96 | 81.2% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 68.04 | 73.01 | 93.2% |
| m=4096, n=2048, k=64 | Attention Score | 12.13 | 11.15 | 108.8% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 34.66 | 41.29 | 83.9% |
| m=4096, n=16384, k=4096 | FFN First Layer | 73.95 | 71.74 | 103.1% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 64.65 | 53.20 | 121.5% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 54.13 | 50.14 | 108.0% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 74.31 | 73.02 | 101.8% |
| m=32768, n=4096, k=4096 | Long Context Processing | 73.27 | 76.78 | 95.4% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 71.83 | 58.35 | 123.1% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:1024}/manual_time       0.651 ms        0.660 ms         1065 bytes_per_second=72.0057Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:8192,k:1024}/manual_time        2.02 ms         1.99 ms          306 bytes_per_second=77.4533Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:2048,k:64}/manual_time         0.088 ms        0.090 ms         7973 bytes_per_second=185.341Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:4096,k:128}/manual_time        0.248 ms        0.246 ms         2790 bytes_per_second=264.189Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:16384,k:4096}/manual_time       7.44 ms         7.41 ms           97 bytes_per_second=37.8236Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:16384}/manual_time       8.49 ms         8.46 ms           85 bytes_per_second=33.1413Gi/s
{hgemm:kernel_type::wmma_opt_2,m:2048,n:5120,k:5120}/manual_time        1.57 ms         1.54 ms          445 bytes_per_second=56.014Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:5120,k:5120}/manual_time        2.86 ms         2.81 ms          250 bytes_per_second=44.3856Gi/s
{hgemm:kernel_type::wmma_opt_2,m:32768,n:4096,k:4096}/manual_time       15.0 ms         15.0 ms           49 bytes_per_second=35.4602Gi/s
{hgemm:kernel_type::wmma_opt_2,m:65536,n:2048,k:2048}/manual_time       7.61 ms         7.56 ms           93 bytes_per_second=66.6958Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.528 ms        0.514 ms         1367 bytes_per_second=88.7609Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.88 ms         1.85 ms          389 bytes_per_second=83.0948Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.096 ms        0.095 ms         7078 bytes_per_second=170.254Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.208 ms        0.208 ms         3374 bytes_per_second=314.397Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.67 ms         7.66 ms          102 bytes_per_second=36.6805Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.3 ms         10.3 ms           68 bytes_per_second=27.3615Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.70 ms         1.71 ms          429 bytes_per_second=51.7174Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.91 ms         2.94 ms          250 bytes_per_second=43.5858Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.3 ms         14.1 ms           50 bytes_per_second=37.1927Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          9.37 ms         9.49 ms           79 bytes_per_second=54.2139Gi/s
```
