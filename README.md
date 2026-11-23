# cabo_vla

VLA training codebase

## Development

Install the Python dependencies and enable the Git hooks for this checkout:

```sh
uv sync --locked
rustup component add rustfmt clippy
uv run pre-commit install
```

Ruff checks Python errors and import ordering and formats Python code. ty checks
Python types across the whole project using the locked project environment. The
pre-commit hooks also check file hygiene, TOML and YAML syntax, and Rust formatting
and linting.

```sh
uv run ruff check .           # lint Python
uv run ruff check --fix .     # fix imports and other safe lint issues
uv run ruff format .          # format Python
uv run ty check               # type-check Python
cargo fmt --manifest-path rust/Cargo.toml --all
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
uv run pre-commit run --all-files
uv run pytest
```

Hooks run automatically on `git commit` after installation. If a hook fixes a
file, stage the changes and commit again.

GitHub Actions runs the `Ruff` and `ty` checks on pull requests and pushes to
`main`. Ruff checks both lint and formatting without changing files; ty checks the
whole project. Both checks must pass before a pull request can merge into `main`.

The `Tests` workflow runs the Python test suite on CPU with Python 3.12 and the
locked uv dependencies, plus Rust tests on the stable toolchain. It runs on pull
requests, pushes to `main`, merge queue updates, and manual dispatches.

## Qwen3.5 VLA on Modal

With a configured Modal profile, fine-tune on LIBERO and save the model weights and training log:

```sh
uvx --from modal==1.6.4 modal run modal_q35vla.py --epochs 1 --batch-size 8
```

Modal caches the Hugging Face inputs and stores training runs in the
`cabo-vla-q35vla` volume. The training command prints the output directory.

Training saves model weights at step zero, 50%, 75%, and the final step. Modal commits
each snapshot to the volume. Use `--wandb-project cabo-vla` with `WANDB_API_KEY` in
the local environment to track loss in W&B. Add `--eval-checkpoints` to run
LIBERO Spatial asynchronously on each committed snapshot and log overall and
per-task success rates on W&B’s checkpoint-step axis. The rollout details are saved alongside
each checkpoint as JSON.

Use `--initial-weights /data/runs/<run-id>/model.safetensors` to warm-start
a Modal training run from a saved policy with a fresh optimizer. Add
`--checkpoint-every-quarter` to save at 25%, 50%, 75%, and 100% of each epoch.

### LIBERO Spatial evaluation

In a Linux environment with Cabo VLA and `hf-libero>=0.1.5,<0.2.0` installed, run:

```sh
MUJOCO_GL=egl python -m cabo_vla.eval_libero_spatial /path/to/model.safetensors
```

The evaluator runs the first five LIBERO Spatial tasks with two trials per task.
It downloads the LIBERO normalization statistics and Qwen tokenizer by default.
Pass `--stats` and `--tokenizer` to use local copies. Use `--video-dir rollouts`
to record one MP4 per trial with both camera views. Add `--temporal-decay 0.5`
to average overlapping arm predictions, halving each plan’s weight per replan interval.
The gripper command comes from the newest plan.


## todo

The goal right now is building an MVP:
- we can load a dataset from disk
- we can train a simple baseline policy (diffusion or patch policy)
- we can inference this policy on a benchmark
- deadline EOM October 2026

### data
we want to completely remove dependancies on lerobot dataloading -- meaning full control of dataloading from our code

### training
### inference
