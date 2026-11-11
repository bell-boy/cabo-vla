The benchmark ran on Runpod `cpu3c` pod `4qkckw9jufgo23` in `EUR-IS-1` with 32 GB RAM and 16 vCPUs mapped to eight AMD EPYC 9965 physical cores. The pod was terminated after the run.

The workload used 100 shuffled LIBERO episodes, 16,017 samples per epoch, seed 429, batch size 32, two 224 × 224 cameras, and action horizon 32. The dataset revision was `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. The pod ran the process-based dataloader from the working tree at base commit `050b519` (dataloader file blob `1fe14d4288556b981d6f75113bad9d32a2890d06`).

The remote benchmark copy set `DataloaderConfig.container_pool_size=80` and emitted each elapsed epoch time to stdout. The benchmark now exposes that setting as a flag. With the dataset staged at `/workspace/libero`, reproduce the run with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu TQDM_DISABLE=1 \
  python -u benchmarks/bench_dataloader.py \
  --path /workspace/libero --workers 1 2 4 8 --container-pool-size 80 --n-epochs 3 --output /tmp/scaling.json
```

| Workers | Epoch times (s) | Samples/s | Speedup | Ideal scaling |
|---:|---|---:|---:|---:|
| 1 | 126.31, 134.60, 132.22 | 122.23 | 1.00× | 100.0% |
| 2 | 68.01, 66.84, 68.84 | 235.90 | 1.93× | 96.5% |
| 4 | 38.64, 36.41, 36.36 | 431.35 | 3.53× | 88.2% |
| 8 | 23.03, 23.88, 24.24 | 675.36 | 5.53× | 69.1% |

Throughput is total samples divided by total epoch time, including worker startup and shutdown. Speedup is relative to one worker. The percentage of ideal scaling divides speedup by worker count. [scaling.json](scaling.json) preserves the streamed per-epoch measurements.
