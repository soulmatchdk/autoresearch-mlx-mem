# UCMD V3.6 Runbook

Status: baseline frozen, benchmark-ready, adapter-clean.

This repo now has one canonical baseline and one canonical optimization loop:

- `train.py` is the frozen UCMD V3.6 baseline
- `run_variant.py` is the locked keep-or-discard loop
- `results.tsv` is append-only and is the run ledger
- LoCoMo is an external benchmark track, not a reason to reopen the baseline

## Core rules

- One run equals one clear hypothesis.
- The agent writes the exact `description`.
- The agent decides `keep` or `discard`.
- The agent records `run_id`, `git_commit`, `git_parent_commit`, `status`, and `decision_reason`.
- Do not delete or rewrite old rows in `results.tsv`.
- Do not auto-merge, auto-rebase, or clean history as part of keep or discard.

## Canonical commands

Regenerate the LoCoMo adapter outputs:

```bash
uv run adapt_locomo.py --input ../data/locomo10.json --output-dir locomo_adapted
```

Inspect the current LoCoMo benchmark slices:

```bash
uv run locomo_breakdown.py --queries locomo_adapted/queries.jsonl --metadata locomo_adapted/metadata.json
```

Run the eval-only LoCoMo benchmark reference:

```bash
uv run locomo_eval.py --events locomo_adapted/events.jsonl --queries locomo_adapted/queries.jsonl --metadata locomo_adapted/metadata.json
```

Run one locked baseline variant:

```bash
uv run run_variant.py --description "one exact single-hypothesis change"
```

If you only want manual inspection without touching the ledger:

```bash
uv run train.py
```

## Keep or discard rule

The loop uses a tolerance of `0.001`.

- `discard` if `hard_joint_answer_or_abstain_acc` goes down
- `discard` if any `regime_mismatch` reappears
- `discard` if `hard_latest_slice_conflict_accuracy` goes down
- `keep` if `hard_joint_answer_or_abstain_acc` goes up
- on ties: prefer higher `hard_latest_slice_recall`
- then prefer higher `hard_latest_slice_conflict_accuracy`
- then prefer higher standard `joint_answer_or_abstain_acc`
- then prefer lower hard retrieval latency

When correctness is tied at `1.0`, small robust wins like lower latency or cleaner external adapter behavior are legitimate `keep` reasons.

## Primary metrics

For the frozen UCMD baseline, optimize in this order:

- `hard_joint_answer_or_abstain_acc`
- `hard_latest_slice_recall`
- `hard_latest_slice_conflict_accuracy`
- standard `joint_answer_or_abstain_acc`
- `hard_avg_retrieval_latency_ms`

Useful but secondary:

- `raw_recall_at_k`
- `header_fake_fact_rate`
- `avg_header_value_purity`

## LoCoMo track

Treat LoCoMo as an external realism-gap benchmark.

Use:

- `observation` as raw events
- `session_summary` as headers
- `qa` as queries
- `session_idx` as the time axis

Use the benchmark track to inspect:

- query mode mix
- category mix
- answerable vs abstain-like questions
- evidence density
- current-query coverage
- eval-only benchmark accuracy per query mode and category
- failure buckets like temporal selection error, missing evidence, false abstain, and multi-hop failure

Do not change the baseline answer rule just because LoCoMo is harder. First inspect the benchmark slices and failure buckets.

## Practical discipline

- If `results.tsv` is the only dirty file, it is okay to run another locked evaluation. The loop now makes an empty `run:<run_id>` commit over the already committed candidate.
- If code changed, commit those exact files through `run_variant.py` so the ledger row points at the tested code.
- If the adapter changes, regenerate the adapter outputs before evaluating LoCoMo again.
