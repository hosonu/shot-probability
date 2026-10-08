# Contributing

This repository predicts whether a football shot becomes a goal. Phase 1 is a logistic regression on a shot table built from StatsBomb open data.

## Setup

Install the local environment from the repository root:

```bash
make install
```

This uses `uv` through the project `Makefile` and creates local generated directories such as `.venv/`, `bin/`, `uv_python/`, and `uv_cache/`.

## Development Workflow

Run the main checks before opening a pull request:

```bash
make lint
make check
```

Format Python files with:

```bash
make format
```

Build the shot table, then fit the logistic baseline:

```bash
make data
make train
```

`make data` writes `data/shots.parquet`. Training reads that file and does not call StatsBomb. Do not commit JSON, the parquet file, `outputs/`, virtual environments, or caches.

Use `make clean` to remove Python caches and saved evaluation files. Use `make fclean` only when you want to remove the local virtual environment, local `uv` install, and generated caches.

## Code Style

- Follow standard Python naming conventions.
- Keep functions and classes focused on one responsibility.
- Prefer scikit-learn-compatible APIs for the model (`Pipeline`, `ClassifierMixin`) when adding another estimator.
- Keep the shot-table schema in `src/shot_probability/columns.py`. Phase 1 features stay `distance_to_goal`, `angle_to_goal`, `body_part`, and `is_set_piece`.
- Do not commit generated model files, downloaded datasets, local caches, or virtual environments.

## Testing and Evaluation

```bash
make check
make train
```

The primary metric is log loss. Record Brier score and the calibration plots. Accuracy is descriptive only. The split unit is `match_id`.

## Branch Naming

Use short, descriptive branch names:

```text
feat/shot-table
fix/goal-angle
docs/update-readme
```

## Commit Messages

Use concise Conventional Commit-style messages:

```text
feat(shots): build the phase-1 shot table
fix(split): keep every match in one partition
docs(readme): clarify the logistic baseline
```

Common types:

- `feat`: user-facing feature or capability
- `fix`: bug fix
- `docs`: documentation-only change
- `refactor`: internal change without behavior change
- `test`: tests or evaluation scripts
- `chore`: tooling, cleanup, or maintenance

## Pull Request Checklist

Before opening a pull request:

- The README and CLI help still match the implemented behavior.
- `make lint` passes.
- `make check` passes, or any failure is explained in the PR.
- New generated files are not committed accidentally.
- Dataset licensing and citation requirements are respected.
