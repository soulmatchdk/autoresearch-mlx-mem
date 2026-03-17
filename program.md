# autoresearch-mlx / UCMD baseline workflow

This repo stays in the `autoresearch-mlx` family: Apple Silicon, native MLX, simple CLI entrypoints, and a single main file for baseline work. The default baseline is now UCMD V3.1 memory in `train.py`.

## Setup

1. Read `README.md` for the current repo identity.
2. Treat `train.py` as the canonical implementation.
3. Use `uv run prepare.py` only as a compatibility helper if you want a reminder about runtime dataset paths.
4. Run `uv run train.py` to train, evaluate, demo the memory API, and append a row to `results.tsv`.

## What you should edit

- Edit `train.py` when changing the baseline.
- Update `README.md`, `program.md`, or `results.tsv` only when the repo story or logged metrics change.
- Avoid introducing new repo files unless a human explicitly asks for them.

## Evaluation target

The old `val_bpb` target is gone. Judge changes by UCMD memory behavior:

- higher joint answer-or-abstain accuracy
- better abstain quality
- stronger raw recall
- lower false merges
- lower unnecessary branching
- few useful headers
- acceptable retrieval latency

## Workflow

The baseline loop is:

1. Observe or synthesize an event stream
2. `add_observation(...)`
3. `retrieve(...)`
4. Answer, abstain, or choose a next action
5. Add outcomes back into memory
6. `consolidate()` periodically

When experimenting on the baseline itself:

1. Change `train.py`
2. Run `uv run train.py`
3. Inspect the printed metrics and demo output
4. Check the appended row in `results.tsv`
5. Keep the change only if it improves the memory system without adding unnecessary complexity

## Design rule

"Everything should be made as simple as possible, but not simpler."

Prefer changes that preserve the one-file MLX baseline, keep headers few and evidence-anchored, and make the memory API more honest rather than more ornate.
