from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from gepa_reasoning_adapter import ReasoningGEPAAdapter
from leakage_red_team import run_leakage_red_team
from locomo_reasoning_eval import render_report
from reasoning_layer_prompts import SEED_CANDIDATE, SMOKE_RUN_CONFIG, candidate_to_json, clone_candidate


LEDGER_PATH = Path("gepa_runs/ledger.tsv")
DECISION_TOLERANCE = 0.001


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GEPA v1 reasoning-layer smoke slice.")
    parser.add_argument("--budget", type=int, default=SMOKE_RUN_CONFIG["eval_budget"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-proposals", type=int, default=SMOKE_RUN_CONFIG["max_proposals"])
    parser.add_argument("--run-dir", default="")
    return parser.parse_args()


def ensure_ledger_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    headers = [
        "run_id",
        "status",
        "budget",
        "seed",
        "leakage_passed",
        "seed_current_joint_acc",
        "seed_temporal_joint_acc",
        "seed_multi_hop_joint_acc",
        "seed_answerable_accuracy",
        "seed_abstain_precision",
        "seed_abstain_recall",
        "seed_false_abstain_penalty",
        "best_current_joint_acc",
        "best_temporal_joint_acc",
        "best_multi_hop_joint_acc",
        "best_answerable_accuracy",
        "best_abstain_precision",
        "best_abstain_recall",
        "best_false_abstain_penalty",
        "best_candidate_name",
        "report_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()


def append_ledger_row(path: Path, row: dict[str, Any]):
    ensure_ledger_header(path)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys(), delimiter="\t")
        writer.writerow(row)


def flatten_summary(summary: dict[str, Any]) -> dict[str, float]:
    metrics = dict(summary["aggregate_metrics"])
    metrics.update(summary["objective_summary"])
    return metrics


def compare_candidates(left: dict[str, float], right: dict[str, float], tolerance: float = DECISION_TOLERANCE) -> int:
    keys = [
        ("current_joint_acc", True),
        ("temporal_joint_acc", True),
        ("answerable_accuracy", True),
        ("abstain_precision", True),
        ("abstain_recall", True),
        ("false_abstain_penalty", False),
        ("multi_hop_joint_acc", True),
        ("false_confident_answer_penalty", False),
    ]
    for key, higher_is_better in keys:
        left_value = float(left.get(key, 0.0))
        right_value = float(right.get(key, 0.0))
        if higher_is_better:
            if left_value > right_value + tolerance:
                return 1
            if left_value < right_value - tolerance:
                return -1
        else:
            if left_value < right_value - tolerance:
                return 1
            if left_value > right_value + tolerance:
                return -1
    return 0


def proposal(name: str, candidate: dict[str, str]) -> dict[str, Any]:
    return {"name": name, "candidate": candidate}


def propose_candidates(seed_candidate: dict[str, str], reflective_dataset: dict[str, list[dict[str, Any]]], max_proposals: int) -> list[dict[str, Any]]:
    proposals = []
    if max_proposals <= 0:
        return proposals

    temporal_candidate = clone_candidate(seed_candidate)
    temporal_candidate["temporal_policy"] += " Extra rule: map recently to the supporting session date whenever the event is grounded."
    temporal_candidate["abstain_policy"] += " Flags: low_grounding_threshold."
    proposals.append(proposal("temporal_recall_tune", temporal_candidate))

    if len(proposals) >= max_proposals:
        return proposals

    current_candidate = clone_candidate(seed_candidate)
    current_candidate["current_policy"] += " Extra rule: prefer attribute-matching snippets and compact value spans. Flags: prefer_attribute_match, prefer_compact_value_span, collection_top6."
    proposals.append(proposal("current_specificity_tune", current_candidate))

    if len(proposals) >= max_proposals:
        return proposals

    multihop_candidate = clone_candidate(seed_candidate)
    multihop_candidate["multi_hop_policy"] += " Extra rule: when counterfactual evidence is thin, still combine the top three grounded snippets before abstaining."
    multihop_candidate["abstain_policy"] += " Extra rule: avoid abstaining on answerable multi-hop cases when there is one grounded synthesis path."
    proposals.append(proposal("multihop_balance_tune", multihop_candidate))

    if reflective_dataset.get("explanation_policy") and len(proposals) < max_proposals:
        explanation_candidate = clone_candidate(seed_candidate)
        explanation_candidate["explanation_policy"] += " Extra rule: always echo one distinctive support term in the explanation."
        proposals.append(proposal("explanation_audit_tune", explanation_candidate))

    return proposals[:max_proposals]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [len(header) for header in headers]
    string_rows = [[str(cell) for cell in row] for row in rows]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [fmt(headers), divider]
    lines.extend(fmt(row) for row in string_rows)
    return "\n".join(lines)


def render_smoke_report(run_id: str, budget: int, leakage_report: dict[str, Any], seed_summary: dict[str, Any], proposal_summaries: list[dict[str, Any]], best_name: str, status: str) -> str:
    lines = [
        "# GEPA Reasoning V1 Smoke Run",
        "",
        f"- run_id: `{run_id}`",
        f"- budget: `{budget}`",
        f"- status: `{status}`",
        f"- leakage_passed: `{leakage_report['passed']}`",
        "",
        "## Candidate Comparison",
        "",
    ]
    rows = []
    all_summaries = [{"name": "seed", "summary": seed_summary}] + proposal_summaries
    for item in all_summaries:
        flat = flatten_summary(item["summary"])
        rows.append(
            [
                item["name"],
                f"{flat['current_joint_acc']:.4f}",
                f"{flat['temporal_joint_acc']:.4f}",
                f"{flat['multi_hop_joint_acc']:.4f}",
                f"{flat['answerable_accuracy']:.4f}",
                f"{flat['abstain_precision']:.4f}",
                f"{flat['abstain_recall']:.4f}",
                f"{flat['false_abstain_penalty']:.4f}",
                f"{flat['false_confident_answer_penalty']:.4f}",
            ]
        )
    lines.append(
        markdown_table(
            [
                "candidate",
                "current",
                "temporal",
                "multi_hop",
                "answerable_acc",
                "abstain_precision",
                "abstain_recall",
                "false_abstain",
                "false_confident",
            ],
            rows,
        )
    )
    lines.extend(["", f"Best candidate: `{best_name}`", "", "## Seed Eval", "", render_report(seed_summary), ""])
    return "\n".join(lines)


def main():
    args = parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("gepa_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = ReasoningGEPAAdapter(seed=args.seed)
    leakage_report = run_leakage_red_team(adapter, dict(SEED_CANDIDATE), budget=min(24, args.budget), seed=args.seed)
    (run_dir / "leakage_red_team.json").write_text(json.dumps(leakage_report, indent=2, ensure_ascii=True) + "\n")

    batch = adapter.sample_batch(budget=args.budget, seed=args.seed)
    seed_eval = adapter.evaluate(batch=batch, candidate=dict(SEED_CANDIDATE), capture_traces=True)
    seed_summary = seed_eval.aggregate_metrics
    reflective_dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), seed_eval, list(SEED_CANDIDATE.keys()))
    (run_dir / "seed_candidate.json").write_text(candidate_to_json(dict(SEED_CANDIDATE)) + "\n")
    (run_dir / "reflective_dataset.json").write_text(json.dumps(reflective_dataset, indent=2, ensure_ascii=True) + "\n")

    proposal_summaries = []
    best_name = "seed"
    best_summary = seed_summary
    best_candidate = dict(SEED_CANDIDATE)
    status = "benchmark_candidate"

    if leakage_report["passed"]:
        for item in propose_candidates(dict(SEED_CANDIDATE), reflective_dataset, args.max_proposals):
            proposal_eval = adapter.evaluate(batch=batch, candidate=item["candidate"], capture_traces=True)
            proposal_summary = proposal_eval.aggregate_metrics
            proposal_summaries.append({"name": item["name"], "summary": proposal_summary})
            (run_dir / f"{item['name']}.json").write_text(candidate_to_json(item["candidate"]) + "\n")
            if compare_candidates(flatten_summary(proposal_summary), flatten_summary(best_summary)) > 0:
                best_name = item["name"]
                best_summary = proposal_summary
                best_candidate = item["candidate"]
        status = "benchmark_win" if best_name != "seed" else "benchmark_candidate"
    else:
        status = "benchmark_discard"

    report = render_smoke_report(run_id, args.budget, leakage_report, seed_summary, proposal_summaries, best_name, status)
    (run_dir / "report.md").write_text(report + "\n")
    (run_dir / "best_candidate.json").write_text(candidate_to_json(best_candidate) + "\n")
    (run_dir / "best_summary.json").write_text(json.dumps(best_summary, indent=2, ensure_ascii=True) + "\n")

    append_ledger_row(
        LEDGER_PATH,
        {
            "run_id": run_id,
            "status": status,
            "budget": args.budget,
            "seed": args.seed,
            "leakage_passed": int(leakage_report["passed"]),
            "seed_current_joint_acc": f"{flatten_summary(seed_summary)['current_joint_acc']:.6f}",
            "seed_temporal_joint_acc": f"{flatten_summary(seed_summary)['temporal_joint_acc']:.6f}",
            "seed_multi_hop_joint_acc": f"{flatten_summary(seed_summary)['multi_hop_joint_acc']:.6f}",
            "seed_answerable_accuracy": f"{flatten_summary(seed_summary)['answerable_accuracy']:.6f}",
            "seed_abstain_precision": f"{flatten_summary(seed_summary)['abstain_precision']:.6f}",
            "seed_abstain_recall": f"{flatten_summary(seed_summary)['abstain_recall']:.6f}",
            "seed_false_abstain_penalty": f"{flatten_summary(seed_summary)['false_abstain_penalty']:.6f}",
            "best_current_joint_acc": f"{flatten_summary(best_summary)['current_joint_acc']:.6f}",
            "best_temporal_joint_acc": f"{flatten_summary(best_summary)['temporal_joint_acc']:.6f}",
            "best_multi_hop_joint_acc": f"{flatten_summary(best_summary)['multi_hop_joint_acc']:.6f}",
            "best_answerable_accuracy": f"{flatten_summary(best_summary)['answerable_accuracy']:.6f}",
            "best_abstain_precision": f"{flatten_summary(best_summary)['abstain_precision']:.6f}",
            "best_abstain_recall": f"{flatten_summary(best_summary)['abstain_recall']:.6f}",
            "best_false_abstain_penalty": f"{flatten_summary(best_summary)['false_abstain_penalty']:.6f}",
            "best_candidate_name": best_name,
            "report_path": str((run_dir / 'report.md').as_posix()),
        },
    )

    print(report)


if __name__ == "__main__":
    main()
