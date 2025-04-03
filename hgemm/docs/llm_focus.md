# LLM-Focused Benchmarks

The latest HGEMM implementations (`wmma_opt_2` and `wmma_opt_3`) have been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my implementations against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_2` (TFLOPs/s) | `wmma_opt_3` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` | `wmma_opt_3`/`rocBLAS` |
|------------------|----------------|-----------------|-----------------|-------------------|----------|----------|
| m=4096, n=4096, k=1024 | QKV Projection | 53.22 | 55.31 | 66.38 | 80.2% | 83.3% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 68.95 | 67.92 | 73.06 | 94.4% | 93.0% |
| m=4096, n=2048, k=64 | Attention Score | 11.23 | 11.54 | 12.77 | 87.9% | 90.3% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 32.25 | 33.04 | 42.47 | 75.9% | 77.8% |
| m=4096, n=16384, k=4096 | FFN First Layer | 70.04 | 77.21 | 76.21 | 91.9% | 101.3% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 67.32 | 67.15 | 54.03 | 124.6% | 124.3% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 50.27 | 64.95 | 62.22 | 80.8% | 104.4% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 75.04 | 77.86 | 73.85 | 101.6% | 105.4% |
| m=32768, n=4096, k=4096 | Long Context Processing | 71.82 | 78.05 | 75.23 | 95.5% | 103.7% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 69.14 | 78.97 | 61.51 | 112.4% | 128.4% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:1024}/manual_time       0.646 ms        0.643 ms         1070 TFLOPS=53.2248 bytes_per_second=72.5428Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:8192,k:1024}/manual_time        2.00 ms         1.95 ms          369 TFLOPS=68.9505 bytes_per_second=78.2877Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:2048,k:64}/manual_time         0.099 ms        0.100 ms         7001 TFLOPS=11.2293 bytes_per_second=165.166Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:4096,k:128}/manual_time        0.267 ms        0.272 ms         2470 TFLOPS=32.246 bytes_per_second=244.651Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:16384,k:4096}/manual_time       7.85 ms         7.81 ms           90 TFLOPS=70.0362 bytes_per_second=35.8241Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:16384}/manual_time       8.17 ms         7.90 ms           89 TFLOPS=67.3235 bytes_per_second=34.418Gi/s
{hgemm:kernel_type::wmma_opt_2,m:2048,n:5120,k:5120}/manual_time        2.18 ms         2.17 ms          388 TFLOPS=50.2741 bytes_per_second=40.3155Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:5120,k:5120}/manual_time        2.87 ms         2.90 ms          253 TFLOPS=75.0383 bytes_per_second=44.3073Gi/s
{hgemm:kernel_type::wmma_opt_2,m:32768,n:4096,k:4096}/manual_time       15.3 ms         15.6 ms           47 TFLOPS=71.8164 bytes_per_second=34.6761Gi/s
{hgemm:kernel_type::wmma_opt_2,m:65536,n:2048,k:2048}/manual_time       7.95 ms         7.72 ms           87 TFLOPS=69.1425 bytes_per_second=63.8622Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:1024}/manual_time       0.622 ms        0.627 ms         1097 TFLOPS=55.3101 bytes_per_second=75.3947Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:8192,k:1024}/manual_time        2.03 ms         2.02 ms          364 TFLOPS=67.919 bytes_per_second=77.1206Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:2048,k:64}/manual_time         0.094 ms        0.095 ms         7433 TFLOPS=11.5409 bytes_per_second=174.161Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:4096,k:128}/manual_time        0.263 ms        0.267 ms         2636 TFLOPS=33.0387 bytes_per_second=248.453Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:16384,k:4096}/manual_time       7.13 ms         7.21 ms          104 TFLOPS=77.2077 bytes_per_second=39.449Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:16384}/manual_time       8.19 ms         8.35 ms           88 TFLOPS=67.1453 bytes_per_second=34.3258Gi/s
{hgemm:kernel_type::wmma_opt_3,m:2048,n:5120,k:5120}/manual_time        1.71 ms         1.72 ms          510 TFLOPS=64.9486 bytes_per_second=51.5309Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:5120,k:5120}/manual_time        2.76 ms         2.77 ms          265 TFLOPS=77.8632 bytes_per_second=45.9715Gi/s
{hgemm:kernel_type::wmma_opt_3,m:32768,n:4096,k:4096}/manual_time       14.1 ms         14.2 ms           53 TFLOPS=78.0459 bytes_per_second=37.6681Gi/s
{hgemm:kernel_type::wmma_opt_3,m:65536,n:2048,k:2048}/manual_time       6.97 ms         6.98 ms          103 TFLOPS=78.972 bytes_per_second=72.8994Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.520 ms        0.527 ms         1393 TFLOPS=66.3834 bytes_per_second=90.2004Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.88 ms         1.89 ms          389 TFLOPS=73.0612 bytes_per_second=82.9546Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.085 ms        0.086 ms         8178 TFLOPS=12.7742 bytes_per_second=191.811Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.204 ms        0.206 ms         3407 TFLOPS=42.4665 bytes_per_second=321.307Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.22 ms         7.12 ms          101 TFLOPS=76.2079 bytes_per_second=38.9334Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.2 ms         10.3 ms           67 TFLOPS=54.0299 bytes_per_second=27.6047Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.78 ms         1.78 ms          491 TFLOPS=62.2228 bytes_per_second=49.3982Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.91 ms         2.79 ms          252 TFLOPS=73.8513 bytes_per_second=43.5944Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.6 ms         14.7 ms           50 TFLOPS=75.2298 bytes_per_second=36.2935Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.94 ms         8.61 ms           78 TFLOPS=61.5102 bytes_per_second=56.8132Gi/s
```