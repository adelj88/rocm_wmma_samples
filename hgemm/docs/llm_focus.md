# LLM-Focused Benchmarks

The latest HGEMM implementations (`wmma_opt_3` and `wmma_opt_4`) have been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my implementations against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_3` (TFLOPs/s) | `wmma_opt_4` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_3`/`rocBLAS` | `wmma_opt_4`/`rocBLAS` |
|------------------|----------------|-----------------|-----------------|-------------------|----------|----------|
| m=4096, n=4096, k=1024 | QKV Projection | 52.50 | 56.29 | 65.82 | 79.8% | 85.5% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 67.91 | 71.22 | 72.58 | 93.6% | 98.1% |
| m=4096, n=2048, k=64 | Attention Score | 11.24 | 12.33 | 12.56 | 89.5% | 98.2% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 33.54 | 38.58 | 40.90 | 82.0% | 94.3% |
| m=4096, n=16384, k=4096 | FFN First Layer | 76.91 | 79.70 | 75.35 | 102.1% | 105.8% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 67.09 | 68.94 | 52.99 | 126.6% | 130.1% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 74.83 | 79.05 | 72.66 | 103.0% | 108.8% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 77.94 | 80.20 | 74.02 | 105.3% | 108.3% |
| m=32768, n=4096, k=4096 | Long Context Processing | 78.15 | 79.70 | 76.46 | 102.2% | 104.2% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 78.90 | 80.68 | 61.95 | 127.4% | 130.2% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:1024}/manual_time       0.657 ms        0.667 ms         1030 TFLOPS=52.5009 bytes_per_second=71.3901Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:8192,k:1024}/manual_time        2.03 ms         2.04 ms          367 TFLOPS=67.909 bytes_per_second=77.0907Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:2048,k:64}/manual_time         0.097 ms        0.098 ms         7050 TFLOPS=11.2412 bytes_per_second=168.067Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:4096,k:128}/manual_time        0.259 ms        0.263 ms         2736 TFLOPS=33.5363 bytes_per_second=252.987Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:16384,k:4096}/manual_time       7.16 ms         7.20 ms          102 TFLOPS=76.9114 bytes_per_second=39.3012Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:16384}/manual_time       8.20 ms         8.26 ms           87 TFLOPS=67.0945 bytes_per_second=34.3043Gi/s
{hgemm:kernel_type::wmma_opt_3,m:2048,n:5120,k:5120}/manual_time        1.44 ms         1.41 ms          508 TFLOPS=74.8332 bytes_per_second=61.1759Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:5120,k:5120}/manual_time        2.76 ms         2.81 ms          261 TFLOPS=77.9402 bytes_per_second=46.0223Gi/s
{hgemm:kernel_type::wmma_opt_3,m:32768,n:4096,k:4096}/manual_time       14.1 ms         14.1 ms           52 TFLOPS=78.1458 bytes_per_second=37.7182Gi/s
{hgemm:kernel_type::wmma_opt_3,m:65536,n:2048,k:2048}/manual_time       6.97 ms         6.89 ms          102 TFLOPS=78.9032 bytes_per_second=72.85Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:4096,k:1024}/manual_time       0.612 ms        0.619 ms         1136 TFLOPS=56.2939 bytes_per_second=76.6163Gi/s
{hgemm:kernel_type::wmma_opt_4,m:8192,n:8192,k:1024}/manual_time        1.93 ms         1.88 ms          383 TFLOPS=71.2183 bytes_per_second=80.8422Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:2048,k:64}/manual_time         0.089 ms        0.090 ms         7833 TFLOPS=12.332 bytes_per_second=183.134Gi/s
{hgemm:kernel_type::wmma_opt_4,m:8192,n:4096,k:128}/manual_time        0.226 ms        0.218 ms         3150 TFLOPS=38.5833 bytes_per_second=289.333Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:16384,k:4096}/manual_time       6.91 ms         6.86 ms          107 TFLOPS=79.6968 bytes_per_second=40.7223Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:4096,k:16384}/manual_time       7.98 ms         7.99 ms           88 TFLOPS=68.943 bytes_per_second=35.2546Gi/s
{hgemm:kernel_type::wmma_opt_4,m:2048,n:5120,k:5120}/manual_time        1.36 ms         1.38 ms          544 TFLOPS=79.0533 bytes_per_second=64.6147Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:5120,k:5120}/manual_time        2.68 ms         2.46 ms          273 TFLOPS=80.1953 bytes_per_second=47.3281Gi/s
{hgemm:kernel_type::wmma_opt_4,m:32768,n:4096,k:4096}/manual_time       13.8 ms         13.9 ms           54 TFLOPS=79.7002 bytes_per_second=38.4704Gi/s
{hgemm:kernel_type::wmma_opt_4,m:65536,n:2048,k:2048}/manual_time       6.82 ms         6.86 ms           98 TFLOPS=80.6823 bytes_per_second=74.4646Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.525 ms        0.530 ms         1355 TFLOPS=65.8179 bytes_per_second=89.3657Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.90 ms         1.90 ms          387 TFLOPS=72.5753 bytes_per_second=82.3926Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.087 ms        0.087 ms         7913 TFLOPS=12.5563 bytes_per_second=187.82Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.214 ms        0.212 ms         3312 TFLOPS=40.8955 bytes_per_second=306.076Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.31 ms         7.27 ms          101 TFLOPS=75.349 bytes_per_second=38.497Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.4 ms         10.3 ms           65 TFLOPS=52.9931 bytes_per_second=27.0835Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.48 ms         1.48 ms          496 TFLOPS=72.6603 bytes_per_second=59.3638Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.91 ms         2.94 ms          250 TFLOPS=74.0173 bytes_per_second=43.697Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.4 ms         14.1 ms           52 TFLOPS=76.4616 bytes_per_second=36.8955Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.88 ms         8.70 ms           79 TFLOPS=61.9471 bytes_per_second=57.2182Gi/s
```
