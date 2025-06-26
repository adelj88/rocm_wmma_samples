# LLM-Focused Benchmarks

The latest HGEMM implementations (`wmma_opt_3`, `wmma_opt_4`, and `wmma_opt_5`) have been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my implementations against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_3` (TFLOPs/s) | `wmma_opt_4` (TFLOPs/s) | `wmma_opt_5` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_3`/`rocBLAS` | `wmma_opt_4`/`rocBLAS` | `wmma_opt_5`/`rocBLAS` |
|------------------|----------------|-----------------|-----------------|-----------------|-------------------|----------|----------|----------|
| m=4096, n=4096, k=1024 | QKV Projection | 55.61 | 58.63 | 66.35 | 70.41 | 79.0% | 83.3% | 94.2% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 69.57 | 73.62 | 74.28 | 74.87 | 92.9% | 98.3% | 99.2% |
| m=4096, n=2048, k=64 | Attention Score | 9.23 | 8.77 | 10.20 | 12.65 | 73.0% | 69.3% | 80.6% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 31.39 | 33.95 | 36.61 | 40.27 | 77.9% | 84.3% | 90.9% |
| m=4096, n=16384, k=4096 | FFN First Layer | 74.93 | 78.35 | 76.78 | 76.56 | 97.9% | 102.3% | 100.3% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 66.52 | 68.69 | 74.41 | 53.73 | 123.8% | 127.9% | 138.5% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 80.67 | 83.68 | 82.04 | 75.29 | 107.1% | 111.2% | 109.0% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 81.72 | 84.80 | 84.00 | 75.71 | 107.9% | 112.0% | 110.9% |
| m=32768, n=4096, k=4096 | Long Context Processing | 76.84 | 79.97 | 80.08 | 76.78 | 100.1% | 104.2% | 104.3% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 73.79 | 77.32 | 77.39 | 61.51 | 120.0% | 125.8% | 125.8% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:1024}/manual_time       0.638 ms        0.702 ms         1093 TFLOPS=55.6067 bytes_per_second=73.5154Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:8192,k:1024}/manual_time        1.98 ms         2.09 ms          372 TFLOPS=69.5657 bytes_per_second=78.9478Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:2048,k:64}/manual_time         0.118 ms        0.191 ms         5851 TFLOPS=9.22987 bytes_per_second=138.928Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:4096,k:128}/manual_time        0.275 ms        0.341 ms         2332 TFLOPS=31.3942 bytes_per_second=238.206Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:16384,k:4096}/manual_time       7.35 ms         7.82 ms          102 TFLOPS=74.9258 bytes_per_second=38.2677Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:16384}/manual_time       8.27 ms         8.79 ms           86 TFLOPS=66.5151 bytes_per_second=33.9959Gi/s
{hgemm:kernel_type::wmma_opt_3,m:2048,n:5120,k:5120}/manual_time        1.33 ms         1.44 ms          526 TFLOPS=80.6712 bytes_per_second=65.9865Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:5120,k:5120}/manual_time        2.63 ms         2.85 ms          271 TFLOPS=81.7155 bytes_per_second=48.2454Gi/s
{hgemm:kernel_type::wmma_opt_3,m:32768,n:4096,k:4096}/manual_time       14.3 ms         14.9 ms           51 TFLOPS=76.8395 bytes_per_second=37.078Gi/s
{hgemm:kernel_type::wmma_opt_3,m:65536,n:2048,k:2048}/manual_time       7.46 ms         7.92 ms           99 TFLOPS=73.7889 bytes_per_second=68.0423Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:4096,k:1024}/manual_time       0.606 ms        0.676 ms         1150 TFLOPS=58.629 bytes_per_second=77.2949Gi/s
{hgemm:kernel_type::wmma_opt_4,m:8192,n:8192,k:1024}/manual_time        1.87 ms         2.00 ms          391 TFLOPS=73.6212 bytes_per_second=83.5879Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:2048,k:64}/manual_time         0.135 ms        0.197 ms         5592 TFLOPS=8.77254 bytes_per_second=121.284Gi/s
{hgemm:kernel_type::wmma_opt_4,m:8192,n:4096,k:128}/manual_time        0.263 ms        0.334 ms         2750 TFLOPS=33.9454 bytes_per_second=248.579Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:16384,k:4096}/manual_time       7.03 ms         7.32 ms          105 TFLOPS=78.3519 bytes_per_second=40.0295Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:4096,k:16384}/manual_time       8.01 ms         8.67 ms           90 TFLOPS=68.6858 bytes_per_second=35.1106Gi/s
{hgemm:kernel_type::wmma_opt_4,m:2048,n:5120,k:5120}/manual_time        1.28 ms         1.39 ms          540 TFLOPS=83.6814 bytes_per_second=68.4426Gi/s
{hgemm:kernel_type::wmma_opt_4,m:4096,n:5120,k:5120}/manual_time        2.54 ms         3.08 ms          281 TFLOPS=84.7971 bytes_per_second=50.0108Gi/s
{hgemm:kernel_type::wmma_opt_4,m:32768,n:4096,k:4096}/manual_time       13.8 ms         14.5 ms           54 TFLOPS=79.973 bytes_per_second=38.5968Gi/s
{hgemm:kernel_type::wmma_opt_4,m:65536,n:2048,k:2048}/manual_time       7.12 ms         7.60 ms          104 TFLOPS=77.3196 bytes_per_second=71.331Gi/s
{hgemm:kernel_type::wmma_opt_5,m:4096,n:4096,k:1024}/manual_time       0.519 ms        0.603 ms         1339 TFLOPS=66.3484 bytes_per_second=90.3061Gi/s
{hgemm:kernel_type::wmma_opt_5,m:8192,n:8192,k:1024}/manual_time        1.85 ms         2.09 ms          378 TFLOPS=74.2825 bytes_per_second=84.27Gi/s
{hgemm:kernel_type::wmma_opt_5,m:4096,n:2048,k:64}/manual_time         0.114 ms        0.184 ms         6383 TFLOPS=10.1988 bytes_per_second=143.901Gi/s
{hgemm:kernel_type::wmma_opt_5,m:8192,n:4096,k:128}/manual_time        0.235 ms        0.312 ms         2958 TFLOPS=36.6094 bytes_per_second=277.908Gi/s
{hgemm:kernel_type::wmma_opt_5,m:4096,n:16384,k:4096}/manual_time       7.17 ms         7.76 ms          103 TFLOPS=76.7819 bytes_per_second=39.2102Gi/s
{hgemm:kernel_type::wmma_opt_5,m:4096,n:4096,k:16384}/manual_time       7.40 ms         8.09 ms           99 TFLOPS=74.4081 bytes_per_second=38.0121Gi/s
{hgemm:kernel_type::wmma_opt_5,m:2048,n:5120,k:5120}/manual_time        1.31 ms         1.44 ms          531 TFLOPS=82.0418 bytes_per_second=67.1144Gi/s
{hgemm:kernel_type::wmma_opt_5,m:4096,n:5120,k:5120}/manual_time        2.56 ms         3.13 ms          271 TFLOPS=83.995 bytes_per_second=49.565Gi/s
{hgemm:kernel_type::wmma_opt_5,m:32768,n:4096,k:4096}/manual_time       13.7 ms         14.4 ms           54 TFLOPS=80.0788 bytes_per_second=38.6473Gi/s
{hgemm:kernel_type::wmma_opt_5,m:65536,n:2048,k:2048}/manual_time       7.12 ms         7.78 ms          105 TFLOPS=77.3877 bytes_per_second=71.3634Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.489 ms        0.576 ms         1447 TFLOPS=70.4119 bytes_per_second=95.8604Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.84 ms         2.31 ms          387 TFLOPS=74.8727 bytes_per_second=84.7655Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.087 ms        0.154 ms         7944 TFLOPS=12.6526 bytes_per_second=188.166Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.215 ms        0.289 ms         3226 TFLOPS=40.2726 bytes_per_second=304.457Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.20 ms         7.81 ms          103 TFLOPS=76.5596 bytes_per_second=39.0776Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.3 ms         10.9 ms           66 TFLOPS=53.7266 bytes_per_second=27.4338Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.43 ms         1.54 ms          511 TFLOPS=75.2869 bytes_per_second=61.5237Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.85 ms         3.44 ms          254 TFLOPS=75.7139 bytes_per_second=44.6167Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.3 ms         15.0 ms           51 TFLOPS=76.7824 bytes_per_second=37.0482Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.94 ms         9.40 ms           78 TFLOPS=61.5072 bytes_per_second=56.795Gi/s
```
