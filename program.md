# autoresearch-mlx / UCMD V3.6 workflow

This repo stays in the `autoresearch-mlx` family: Apple Silicon, native MLX, simple CLI entrypoints, and one main file for baseline work. The frozen baseline is now UCMD V3.6 in `train.py`.

## Baseline spec

UCMD V3.6 should be understood as:

- append-only raw memory
- scope-aware retrieval
- freshness-first final answer selection for `current` queries
- headers as secondary retrieval/compression structures
- abstain when the latest relevant slice is missing or conflicted

Do not treat headers as the source of truth. They are useful retrieval summaries, but the final answer path is grounded in raw evidence.

## Setup

1. Read `README.md` for the repo identity and current baseline story.
2. Treat `train.py` as the canonical implementation.
3. Use `uv run prepare.py` only as a compatibility helper if you want a reminder about runtime dataset paths.
4. Run `uv run run_variant.py --description "exact single-hypothesis change"` to auto-commit, train, evaluate, compare against the best kept run, and append a row to the repo-local `results.tsv`.
5. Use `RUNBOOK.md` when you want the operational version of the workflow rather than the rationale.

## What you should edit

- Edit `train.py` when testing a new baseline variant.
- Update `README.md` or `program.md` only when the frozen baseline story or workflow changes.
- Keep `results.tsv` intact. It is the shared run ledger used to compare variants over time, including commit id, parent commit, keep or discard choice, exact change description, and the agent's decision reason.

## Evaluation target

The old `val_bpb` target is gone. Judge changes by UCMD behavior, in this order:

- higher joint answer-or-abstain accuracy
- stronger abstain precision and recall
- higher `latest_slice_recall`
- higher `latest_slice_conflict_accuracy`
- preserved or improved `raw_recall_at_k`
- fewer wrong answers without reintroducing regime drift
- acceptable retrieval latency

Header metrics remain useful health signals, but they are no longer headline success criteria for the baseline.

The locked keep or discard rule uses a tolerance of `0.001`:

- `discard` if `hard_joint_answer_or_abstain_acc` goes down
- `discard` if any `regime_mismatch` reappears
- `discard` if `hard_latest_slice_conflict_accuracy` goes down
- `keep` if `hard_joint_answer_or_abstain_acc` goes up
- on ties: prefer higher `hard_latest_slice_recall`, then higher `hard_latest_slice_conflict_accuracy`, then higher standard `joint_answer_or_abstain_acc`, then lower hard latency

## Workflow

When iterating on the baseline itself:

1. Change `train.py`
2. Make sure the diff reflects one clear hypothesis
3. Run `uv run run_variant.py --description "exact change"`
4. Inspect the printed standard eval, hard eval, and demo output
5. Check the appended row in `results.tsv`, including `git_commit`, `git_parent_commit`, `status`, and `decision_reason`
6. Keep iterating only from the new `keep` reference, never by deleting earlier history

When creating new variants after V3.6, optimize against `results.tsv` rather than memory or intuition. The run log is the source of truth for whether a new idea is actually winning.

`uv run train.py` remains available for raw manual inspection, but it is not the canonical ledger-writing path anymore.

## External benchmark track

LoCoMo is now an external benchmark and realism-gap track, not a reason to reopen the frozen baseline.

Use:

```bash
uv run adapt_locomo.py --input ../data/locomo10.json --output-dir locomo_adapted
uv run locomo_breakdown.py --queries locomo_adapted/queries.jsonl --metadata locomo_adapted/metadata.json
uv run locomo_eval.py --events locomo_adapted/events.jsonl --queries locomo_adapted/queries.jsonl --metadata locomo_adapted/metadata.json
```

Inspect at least:

- query mode mix
- category mix
- answerable vs abstain-like questions
- evidence density and no-evidence slices
- eval-only benchmark accuracy by query mode and category
- failure buckets such as temporal selection error, missing evidence, false abstain, and multi-hop failure

Do not fold LoCoMo-specific benchmark logic back into `train.py` unless the ledger shows a real winning variant.

## Design rule

"Everything should be made as simple as possible, but not simpler."

Prefer variants that preserve the one-file MLX baseline, keep raw evidence authoritative, and only add structure when the logged results justify it.
