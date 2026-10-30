# uv run --with pyinstrument python benchmarks/profile_dataloader.py --num-workers 8 > profile.html
import argparse
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

from huggingface_hub import snapshot_download
from pyinstrument import Profiler

from cabo_vla.data.dataloader import Dataloader, DataloaderConfig, NormalizationType


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the dataloader and write HTML to stdout")
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=429)
    args = parser.parse_args()
    if args.n_epochs <= 0 or args.num_workers <= 0:
        parser.error("n-epochs and num-workers must be positive")
    return args


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    profiler = Profiler()
    with redirect_stdout(sys.stderr):
        config = DataloaderConfig(
            path=Path(snapshot_download("lerobot/libero", repo_type="dataset")),
            resolution=(224, 224),
            batch_size=32,
            shuffle=True,
            action_features=(("action", NormalizationType.QUANTILE),),
            action_horizon=32,
            observation_features=("observation.images.image", "observation.images.image2"),
            prio_features=(("observation.state", NormalizationType.QUANTILE),),
            seq_len=0,
            episode_indices=tuple(rng.sample(range(1693), 100)),
            num_workers=args.num_workers,
        )
        dataloader = Dataloader(config)
        # Profile iteration, excluding the dataset download and loader setup.
        with profiler:
            for _ in range(args.n_epochs):
                for batch in dataloader:
                    pass
    print(profiler.output_html())


if __name__ == "__main__":
    main()
