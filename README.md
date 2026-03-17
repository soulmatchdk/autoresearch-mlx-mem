# autoresearch-mlx

Apple Silicon (MLX) port in the spirit of Karpathy's `autoresearch`, now frozen around a UCMD V3.6 single-file memory baseline for agent loops on Mac.

The baseline is intentionally simple:

- append-only raw memory is the source of truth
- retrieval is scope-aware
- `current` queries are answered from the freshest relevant evidence slice
- headers are secondary retrieval/compression aids, not authoritative facts
- abstention happens when the latest relevant slice is missing or conflicted

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare.py
uv run run_variant.py --description "one clear hypothesis"
```

`run_variant.py` is now the canonical loop. It auto-commits the current variant with `run:<run_id>`, runs the experiment, compares the new run against the best earlier `keep`, and appends the final decision to the repo-local `results.tsv`.

For day-to-day operations, see `RUNBOOK.md`.

## What matters

- `train.py` - the canonical UCMD V3.6 baseline runner in one MLX-native file
- `run_variant.py` - the locked variant loop with auto-commit and keep or discard decisions
- `RUNBOOK.md` - the practical operating guide for the locked loop, LoCoMo regeneration, and keep rules
- `locomo_breakdown.py` - eval-only LoCoMo slice report by query mode and category
- `locomo_eval.py` - eval-only LoCoMo benchmark report with accuracy and failure buckets
- `gepa_reasoning_v1.py` - official GEPA reasoning-layer smoke runner for the LoCoMo benchmark track
- `prepare.py` - compatibility helper; no mandatory data prep step remains
- `program.md` - workflow for iterating on the frozen baseline and logging variants
- `results.tsv` - tab-separated run log and optimization target

## Baseline behavior

The baseline keeps four memory layers:

- `raw_hot` for recent append-only evidence
- `raw_cold` for older append-only evidence behind a scope index
- `headers` for compressed retrieval hints
- `semantic_state` for slow consolidation support

Final answers do not come from a free latent head. They come from evidence-weighted voting over retrieved raw items, with freshness-first selection for `current` queries. Older evidence can still help retrieval, headers, and abstain diagnostics, but it does not overrule the latest relevant slice.

## Metrics

`uv run run_variant.py --description "..."` runs the baseline and logs UCMD-focused metrics, including:

- joint answer-or-abstain accuracy
- abstain accuracy, precision, and recall
- `raw_recall_at_k`
- `latest_slice_recall`
- `latest_slice_conflict_accuracy`
- false merge rate
- unnecessary branch rate
- header health signals
- average retrieval latency

The old raw recall metric is still useful, but for V3.6 the more honest answer-path metrics are `latest_slice_recall` and `latest_slice_conflict_accuracy`.

For the external LoCoMo benchmark track, regenerate the adapter and print the benchmark slices with:

```bash
uv run adapt_locomo.py --input ../data/locomo10.json --output-dir locomo_adapted
uv run locomo_breakdown.py --queries locomo_adapted/queries.jsonl --metadata locomo_adapted/metadata.json
uv run locomo_eval.py --events locomo_adapted/events.jsonl --queries locomo_adapted/queries.jsonl --metadata locomo_adapted/metadata.json
```

For the separate GEPA reasoning slice on top of the frozen kernel:

```bash
uv run gepa_reasoning_v1.py --budget 32 --max-metric-calls 48 --proposal-mode custom
GEPA_REFLECTION_LM="your-model-name" uv run gepa_reasoning_v1.py --budget 32 --max-metric-calls 48 --proposal-mode auto
```

`--proposal-mode auto` now promotes to the official `reflection-LM` path whenever `GEPA_REFLECTION_LM` or `--reflection-lm` is set, and otherwise falls back to the deterministic custom proposer.

Each ledger row now records:

- `run_id`
- `git_commit`
- `git_parent_commit`
- `description`
- `status`
- `decision_reason`
- standard eval metrics
- hard eval metrics

The agent supplies the precise `description`. `status` and `decision_reason` are decided automatically after the run.

## Hard eval

The script also runs a harder synthetic eval split with temporal overrides, latest-slice conflicts, thin latest evidence, missing recent scope, near-regime distractors, and no-scope cases. This is meant to stress the frozen baseline before any new variant is considered a winner.

If you only want to inspect the model without touching the ledger, you can still run `uv run train.py` directly.

## Repo lineage

This remains an `autoresearch-mlx`-style repo: Apple Silicon first, simple entrypoints, and friendly to autonomous agent iteration. The difference is that the default baseline is now an explicit freshness-first agent memory system rather than a language-model pretraining loop.
