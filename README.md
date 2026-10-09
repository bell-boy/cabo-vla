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
