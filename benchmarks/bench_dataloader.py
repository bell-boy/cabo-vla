# Benchmark the dataloader on how long it take to iterate through
# LIBERO's training data.
import argparse
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
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dataloader_config = DataloaderConfig(
        path=Path(snapshot_download("lerobot/libero", repo_type="dataset")),
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
    for num_workers in (1, 2, 4, 8):
        print(f"\nThreads: {num_workers}", flush=True)
        dataloader = Dataloader(replace(dataloader_config, num_workers=num_workers))
        times = []
        for _ in trange(args.n_epochs, desc=f"Epochs ({num_workers} threads)"):
            start_time = time.perf_counter()
            for batch in tqdm(dataloader, desc="Batches", leave=False):
                pass
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        for i, t in enumerate(times):
            print(f"Epoch {i}: {t:.2f} seconds")
        print(f"Total time for {args.n_epochs} epochs: {sum(times):.2f} seconds")
        print(f"Samples per second: {len(dataloader._order_buffer) * args.n_epochs / sum(times):.2f}", flush=True)


if __name__ == "__main__":
    main()
