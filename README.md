# autoresearch-mlx

Apple Silicon (MLX) port in the spirit of Karpathy's `autoresearch`, now shipping a UCMD V3.1 single-file memory baseline for agent loops on Mac.

This repo keeps the same practical shape: native MLX, one main mutable training file, lightweight experiment logging, and an agent-oriented workflow. The default baseline is no longer the old `val_bpb` language-model loop. It is now a structured UCMD memory system implemented directly in [`train.py`](train.py).

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install dependencies
uv sync

# optional compatibility step; explains runtime data paths
uv run prepare.py

# train and evaluate the UCMD baseline
uv run train.py
```

By default the script generates synthetic supervision in memory. If you want a reusable dataset snapshot outside the repo, set `UCMD_DATA=/path/to/ucmd.jsonl` before running.

## What matters

- `train.py` - the canonical UCMD V3.1 baseline in one MLX-native file
- `prepare.py` - compatibility helper; no mandatory data-prep step remains
- `program.md` - agent workflow and experiment protocol for this hybrid repo
- `results.tsv` - tab-separated run log for UCMD metrics

## Baseline behavior

The baseline keeps four memory layers:

- `raw_hot` for recent append-only evidence
- `raw_cold` for older append-only evidence behind a scope index
- `headers` for few soft evidence-backed hypotheses
- `semantic_state` for slow consolidated memory

Training uses only small trainable MLP modules and heads. Field features are fixed random tables. No large backbone, no RL, and no external dataset is required for the default run.

## Metrics

`uv run train.py` prints and logs UCMD-focused metrics instead of `val_bpb`, including:

- joint answer-or-abstain accuracy
- abstain accuracy and precision
- raw recall at `K`
- false merge rate
- unnecessary branch rate
- header fake-fact rate
- average headers used
- average retrieval latency

## Repo lineage

This remains an `autoresearch-mlx`-style repo: Apple Silicon first, simple entrypoints, and friendly to autonomous agent iteration. The difference is that the default baseline is now an explicit agent memory system rather than a language-model pretraining loop.
