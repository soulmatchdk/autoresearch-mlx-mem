from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from gepa_reasoning_adapter import ReasoningGEPAAdapter
from reasoning_layer_prompts import SEED_CANDIDATE


FORBIDDEN_INPUT_KEYS = {"gold_answer", "gold_mode_label", "gold_abstain", "gold_evidence_dia_ids", "category"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a leakage red-team check for the GEPA v1 reasoning-layer slice.")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--write-json", default="")
    return parser.parse_args()


def stable_output_signature(output: dict[str, Any]) -> tuple[Any, ...]:
    return (
        output.get("query_mode"),
        output.get("answer_candidate"),
        bool(output.get("should_abstain")),
        output.get("confidence_band"),
        tuple(item.get("text") for item in output.get("support_items", [])[:2]),
    )


def mutate_batch(batch: list[dict[str, Any]], field: str, value: Any) -> list[dict[str, Any]]:
    mutated = []
    for row in batch:
        new_row = copy.deepcopy(row)
        new_row[field] = copy.deepcopy(value)
        mutated.append(new_row)
    return mutated


def compare_outputs(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return [stable_output_signature(item) for item in left] == [stable_output_signature(item) for item in right]


def scan_for_forbidden_keys(record: Any) -> list[str]:
    hits = []
    if isinstance(record, dict):
        for key, value in record.items():
            if key in FORBIDDEN_INPUT_KEYS:
                hits.append(key)
            hits.extend(scan_for_forbidden_keys(value))
    elif isinstance(record, list):
        for value in record:
            hits.extend(scan_for_forbidden_keys(value))
    return hits


def run_leakage_red_team(adapter: ReasoningGEPAAdapter, candidate: dict[str, str], budget: int = 24, seed: int = 13) -> dict[str, Any]:
    batch = adapter.sample_batch(budget=budget, seed=seed)
    baseline = adapter.evaluate(batch=batch, candidate=candidate, capture_traces=True)
    tests = []

    mutations = [
        ("gold_answer", "__LEAK_TEST_ANSWER__"),
        ("gold_evidence_dia_ids", ["__LEAK_TEST_EVIDENCE__"]),
        ("category", 999),
        ("query_mode", "adversarial"),
    ]
    for field, value in mutations:
        mutated_batch = mutate_batch(batch, field, value)
        mutated = adapter.evaluate(batch=mutated_batch, candidate=candidate, capture_traces=True)
        passed = compare_outputs(baseline.outputs, mutated.outputs)
        tests.append({"name": f"runtime_ignores_{field}", "passed": passed})

    reflective_dataset = adapter.make_reflective_dataset(candidate, baseline, list(candidate.keys()))
    forbidden_hits = []
    for component, rows in reflective_dataset.items():
        for row in rows:
            forbidden_hits.extend(f"{component}:{key}" for key in scan_for_forbidden_keys(row.get("Inputs", {})))
            forbidden_hits.extend(f"{component}:{key}" for key in scan_for_forbidden_keys(row.get("Generated Outputs", {})))
    tests.append({"name": "reflective_inputs_are_gold_free", "passed": not forbidden_hits, "details": forbidden_hits[:20]})

    passed = all(test["passed"] for test in tests)
    return {
        "passed": passed,
        "budget": budget,
        "seed": seed,
        "tests": tests,
    }


def main():
    args = parse_args()
    adapter = ReasoningGEPAAdapter(seed=args.seed)
    report = run_leakage_red_team(adapter, dict(SEED_CANDIDATE), budget=args.budget, seed=args.seed)
    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
