# LLM-Focused Benchmarks

The best HGEMM implementation (`wmma_opt_2`) has been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my best implementation against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_2` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` Ratio |
|------------------|----------------|-----------------|-------------------|-------------------|
| m=4096, n=4096, k=1024 | QKV Projection | 51.73 | 62.09 | 83.3% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 67.75 | 72.23 | 93.8% |
| m=4096, n=2048, k=64 | Attention Score | 8.47 | 9.30 | 91.0% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 29.22 | 36.99 | 79.0% |
| m=4096, n=16384, k=4096 | FFN First Layer | 69.18 | 75.29 | 91.9% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 66.68 | 51.74 | 128.9% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 58.51 | 71.10 | 82.3% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 72.92 | 73.07 | 99.8% |
| m=32768, n=4096, k=4096 | Long Context Processing | 71.55 | 76.30 | 93.8% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 68.04 | 61.88 | 110.0% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:1024}/manual_time       0.668 ms        0.672 ms         1047 TFLOPS=51.729 bytes_per_second=70.1938Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:8192,k:1024}/manual_time        2.03 ms         2.03 ms          361 TFLOPS=67.7491 bytes_per_second=76.9183Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:2048,k:64}/manual_time         0.131 ms        0.132 ms         5091 TFLOPS=8.46654 bytes_per_second=125.026Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:4096,k:128}/manual_time        0.298 ms        0.295 ms         2225 TFLOPS=29.2163 bytes_per_second=219.922Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:16384,k:4096}/manual_time       7.95 ms         7.90 ms           89 TFLOPS=69.176 bytes_per_second=35.3838Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:16384}/manual_time       8.25 ms         8.35 ms           88 TFLOPS=66.6813 bytes_per_second=34.0919Gi/s
{hgemm:kernel_type::wmma_opt_2,m:2048,n:5120,k:5120}/manual_time        1.84 ms         1.80 ms          382 TFLOPS=58.5104 bytes_per_second=47.8413Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:5120,k:5120}/manual_time        2.95 ms         2.91 ms          247 TFLOPS=72.9187 bytes_per_second=43.0576Gi/s
{hgemm:kernel_type::wmma_opt_2,m:32768,n:4096,k:4096}/manual_time       15.4 ms         15.3 ms           47 TFLOPS=71.552 bytes_per_second=34.5466Gi/s
{hgemm:kernel_type::wmma_opt_2,m:65536,n:2048,k:2048}/manual_time       8.08 ms         7.90 ms           85 TFLOPS=68.0446 bytes_per_second=62.8419Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.558 ms        0.552 ms         1330 TFLOPS=62.0852 bytes_per_second=83.9381Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.91 ms         1.89 ms          388 TFLOPS=72.2308 bytes_per_second=81.9526Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.118 ms        0.119 ms         5907 TFLOPS=9.30067 bytes_per_second=138.237Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.236 ms        0.240 ms         3065 TFLOPS=36.9854 bytes_per_second=277.331Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.31 ms         7.43 ms          101 TFLOPS=75.2901 bytes_per_second=38.4594Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.6 ms         10.7 ms           64 TFLOPS=51.74 bytes_per_second=26.4277Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.51 ms         1.50 ms          490 TFLOPS=71.0975 bytes_per_second=58.1181Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.95 ms         3.01 ms          244 TFLOPS=73.0654 bytes_per_second=43.0997Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.4 ms         14.4 ms           51 TFLOPS=76.298 bytes_per_second=36.8211Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.88 ms         8.90 ms           79 TFLOPS=61.8791 bytes_per_second=57.1542Gi/s
```
