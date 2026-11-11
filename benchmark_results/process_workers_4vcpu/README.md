The benchmark ran on Runpod `cpu3c` pod `pl5ty9vlbyrukc` in `EU-RO-1` with Ubuntu 24.04, 8 GB RAM, and four vCPUs on two AMD EPYC 9655P physical cores. The assigned CPUs were 41, 70, 137, and 166. It used 100 shuffled LIBERO episodes (16,017 samples per epoch), seed 429, batch size 32, two cameras at 224 × 224, and action horizon 32.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run --locked --no-dev python -u benchmarks/bench_dataloader.py \
  --path /workspace/libero --workers 1 2 4 --n-epochs 3 --output results/scaling.json
```

| Workers | Epoch times (s) | Samples/s | Speedup |
|---:|---|---:|---:|
| 1 | 107.19, 107.21, 105.41 | 150.25 | 1.00× |
| 2 | 58.71, 56.70, 53.64 | 284.25 | 1.89× |

Throughput divides the total sample count by the sum of the epoch times. `scaling.json` preserves the completed results read from the pod. The last four-worker progress sample showed 100 of 501 batches completed in the first epoch at a recent rate of 12.97 batches/s. The pod became unavailable before the final four-worker result could be collected. Runpod returned HTTP 404 for the pod after the SSH connection failed.
