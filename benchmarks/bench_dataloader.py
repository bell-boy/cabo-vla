# Benchmark the dataloader on how long it take to iterate through
# LIBERO's training data.
import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm, trange

from cabo_vla.data.dataloader import Dataloader, DataloaderConfig, NormalizationType

BATCH_SIZE = 32
TOTAL_LIBERO_EPISODES = 1693


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the dataloader")
    parser.add_argument("--n-epochs", type=int, default=1, help="Number of epochs to run")
    parser.add_argument("--seed", type=int, default=429, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--container-pool-size", type=int, help="Override the decoder cache capacity per worker")
    parser.add_argument("--path", type=Path, help="Use an already downloaded dataset")
    parser.add_argument("--output", type=Path, help="Write benchmark measurements as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dataloader_config = DataloaderConfig(
        path=args.path or Path(snapshot_download("lerobot/libero", repo_type="dataset")),
        resolution=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=True,
        action_features=(("action", NormalizationType.QUANTILE),),
        action_horizon=32,
        observation_features=("observation.images.image", "observation.images.image2"),
        prio_features=(("observation.state", NormalizationType.QUANTILE),),
        seq_len=0,
        episode_indices=tuple(rng.sample(range(TOTAL_LIBERO_EPISODES), 100)),
    )
    if args.container_pool_size is not None:
        dataloader_config = replace(dataloader_config, container_pool_size=args.container_pool_size)
    results = []
    for num_workers in args.workers:
        print(f"\nProcesses: {num_workers}", flush=True)
        dataloader = Dataloader(replace(dataloader_config, num_workers=num_workers))
        times = []
        for _ in trange(args.n_epochs, desc=f"Epochs ({num_workers} processes)"):
            start_time = time.perf_counter()
            for batch in tqdm(dataloader, desc="Batches", leave=False):
                pass
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        results.append({"workers": num_workers, "samples": len(dataloader._order_buffer), "epoch_seconds": times})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        for i, t in enumerate(times):
            print(f"Epoch {i}: {t:.2f} seconds")
        print(f"Total time for {args.n_epochs} epochs: {sum(times):.2f} seconds")
        print(f"Samples per second: {len(dataloader._order_buffer) * args.n_epochs / sum(times):.2f}", flush=True)


if __name__ == "__main__":
    main()
