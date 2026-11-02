# Parallel Gaussian backward atomics: balanced A/B

Measured source `b96db27aa5ef68e67d8c8bc3b3e6fa3a7548dcac`. Ten paired rounds, order reversed each round. All other code paths are unchanged in the same binary. GPU event timing below isolates the synthetic backward launch; real-scene rows use synchronized wall time. The real scene has 139410 Gaussians. Host-load observations are in hostload.log.

| Workload | Metric | Serial atomics: median [min,max] | Parallel atomics: median [min,max] | Change | Lower pairs |
|---|---|---|---|---|---|
| 1024 | backward_median_ms | 26.8997 [25.4211,83.9963] | 26.6936 [24.1562,30.4782] | -0.77% | 7/10 |
| 1024 | cuda_peak_allocated_backward | 232.8652 [232.8652,232.8652] | 232.8652 [232.8652,232.8652] | +0.00% | 0/10 |
| 1024 | cuda_peak_allocated_forward | 129.7222 [129.7222,129.7222] | 129.7222 [129.7222,129.7222] | +0.00% | 0/10 |
| 1024 | cuda_peak_reserved | 308.0000 [308.0000,308.0000] | 308.0000 [308.0000,308.0000] | +0.00% | 0/10 |
| 1024 | forward_backward_median_ms | 41.2364 [38.6464,144.1280] | 40.6898 [37.0810,46.9630] | -1.33% | 5/10 |
| 1024 | forward_median_ms | 12.7130 [12.3628,54.3950] | 13.0558 [12.1170,15.4249] | +2.70% | 7/10 |
| 1024 | process_peak_working_set | 976.2812 [975.8867,976.6211] | 976.2207 [975.7227,976.5586] | -0.01% | 6/10 |
| 1024 | train_median_ms | 66.5819 [65.0083,108.3650] | 67.0880 [63.5465,81.9844] | +0.76% | 6/10 |
| 512 | backward_median_ms | 14.4825 [13.5956,15.0187] | 14.1775 [13.1696,15.8263] | -2.11% | 5/10 |
| 512 | cuda_peak_allocated_backward | 196.5239 [196.5239,196.5239] | 196.5239 [196.5239,196.5239] | +0.00% | 0/10 |
| 512 | cuda_peak_allocated_forward | 100.1484 [100.1484,100.1484] | 100.1484 [100.1484,100.1484] | +0.00% | 0/10 |
| 512 | cuda_peak_reserved | 244.0000 [244.0000,244.0000] | 244.0000 [244.0000,244.0000] | +0.00% | 0/10 |
| 512 | forward_backward_median_ms | 25.1448 [23.1194,26.2427] | 24.0602 [22.3110,27.2032] | -4.31% | 8/10 |
| 512 | forward_median_ms | 9.7597 [8.8484,10.5992] | 10.0796 [8.6266,11.4057] | +3.28% | 3/10 |
| 512 | process_peak_working_set | 965.1914 [964.7969,965.5312] | 965.1309 [964.6328,965.4688] | -0.01% | 6/10 |
| 512 | train_median_ms | 27.3987 [24.8768,110.9190] | 27.3794 [24.8001,40.5510] | -0.07% | 6/10 |
| dense | gpu_ms | 9.8951 [9.6860,10.2337] | 9.5803 [9.1184,15.1032] | -3.18% | 8/10 |
| dense | wall_ms | 10.3823 [10.0374,10.6571] | 10.0152 [9.3436,16.2604] | -3.54% | 8/10 |
| row-groups | gpu_ms | 2.0249 [1.9663,2.1014] | 1.4322 [1.3599,1.9874] | -29.27% | 10/10 |
| row-groups | wall_ms | 2.3995 [2.2769,2.4957] | 1.7960 [1.6379,3.1025] | -25.15% | 9/10 |
| sparse | gpu_ms | 2.3652 [2.1556,2.3961] | 2.1053 [1.9075,3.2451] | -10.99% | 9/10 |
| sparse | wall_ms | 2.7443 [2.4569,2.8098] | 2.4574 [2.2143,4.3636] | -10.46% | 8/10 |

Memory columns are MiB. This branch only records render-phase CUDA peaks, not training VRAM. Working set is cumulative over the complete process. The forward-only measurements are unchanged-path noise controls. Synthetic tile layouts are stress controls, not observed real-scene coverage. Training throughput can be derived as 1000/step_ms; no accuracy or convergence improvement is implied.
