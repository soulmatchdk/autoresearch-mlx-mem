from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from gepa import optimize

from gepa_reasoning_adapter import ReasoningGEPAAdapter
from leakage_red_team import run_leakage_red_team
from locomo_reasoning_eval import render_report
from reasoning_layer_prompts import SEED_CANDIDATE, SMOKE_RUN_CONFIG, candidate_to_json, clone_candidate


LEDGER_PATH = Path("gepa_runs/ledger.tsv")
DECISION_TOLERANCE = 0.001
REFLECTION_LM_ENV = "GEPA_REFLECTION_LM"
LEDGER_HEADERS = [
    "run_id",
    "status",
    "engine_mode",
    "reflection_lm",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GEPA v1 reasoning-layer slice with the official GEPA engine.")
    parser.add_argument("--budget", type=int, default=SMOKE_RUN_CONFIG["eval_budget"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-metric-calls", type=int, default=96)
    parser.add_argument("--reflection-minibatch-size", type=int, default=6)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--proposal-mode", choices=("auto", "custom", "reflection_lm"), default="auto")
    parser.add_argument(
        "--reflection-lm",
        default="",
        help=f"Optional official GEPA reflection LM model name. Falls back to ${REFLECTION_LM_ENV} when unset.",
    )
    return parser.parse_args()


def ensure_ledger_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_HEADERS, delimiter="\t")
            writer.writeheader()
        return

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        existing_headers = reader.fieldnames or []
        if existing_headers == LEDGER_HEADERS:
            return
        rows = list(reader)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_HEADERS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in LEDGER_HEADERS})


def append_ledger_row(path: Path, row: dict[str, Any]):
    ensure_ledger_header(path)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_HEADERS, delimiter="\t")
        writer.writerow({header: row.get(header, "") for header in LEDGER_HEADERS})


def resolve_reflection_lm_name(raw_value: str) -> str:
    direct = raw_value.strip()
    if direct:
        return direct
    return os.environ.get(REFLECTION_LM_ENV, "").strip()


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


def split_train_val(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(batch) < 12:
        return batch, batch
    val_size = max(8, len(batch) // 3)
    valset = batch[:val_size]
    trainset = batch[val_size:]
    if not trainset:
        trainset = valset
    return trainset, valset


def apply_component_tune(candidate: dict[str, str], component: str, iteration: int) -> dict[str, str]:
    tuned = clone_candidate(candidate)
    if component == "temporal_policy":
        tuned["temporal_policy"] += " Extra rule: map recently to the supporting session date whenever the event is grounded."
        tuned["abstain_policy"] += " Flags: low_grounding_threshold."
    elif component == "current_policy":
        tuned["current_policy"] += " Extra rule: prefer attribute-matching snippets and compact value spans. Flags: prefer_attribute_match, prefer_compact_value_span, collection_top6."
    elif component == "multi_hop_policy":
        tuned["multi_hop_policy"] += " Extra rule: combine the top three grounded snippets before giving up on a likely answer."
        tuned["abstain_policy"] += " Extra rule: avoid abstaining on answerable multi-hop cases when there is one grounded synthesis path."
    elif component == "abstain_policy":
        tuned["abstain_policy"] += " Extra rule: lower the grounding threshold when one snippet directly answers the question. Flags: low_grounding_threshold."
    elif component == "answer_synthesis_policy":
        tuned["answer_synthesis_policy"] += " Extra rule: prefer compact grounded value spans over full-sentence restatements."
        tuned["current_policy"] += " Flags: prefer_compact_value_span."
    elif component == "explanation_policy":
        tuned["explanation_policy"] += " Extra rule: always echo one distinctive support term in the explanation."
    elif component == "query_mode_rubric":
        tuned["query_mode_rubric"] += " Extra rule: treat how long/how long ago as temporal and would/if/likely as multi-hop."
    tuned["query_mode_rubric"] += f" Iteration note: custom proposer step {iteration}."
    return tuned


def make_custom_candidate_proposer():
    state = {"calls": 0}

    def proposer(candidate: dict[str, str], reflective_dataset: dict[str, list[dict[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        state["calls"] += 1
        targets = list(components_to_update or [])
        if not targets:
            targets = ["query_mode_rubric"]
        tuned = clone_candidate(candidate)
        for component in targets:
            if component in reflective_dataset or component in tuned:
                tuned = apply_component_tune(tuned, component, state["calls"])
        return tuned

    return proposer


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


def render_smoke_report(
    run_id: str,
    budget: int,
    engine_mode: str,
    reflection_lm_name: str,
    leakage_report: dict[str, Any],
    seed_summary: dict[str, Any],
    best_summary: dict[str, Any],
    best_name: str,
    status: str,
    result_metrics: dict[str, Any],
) -> str:
    rows = []
    for name, summary in (("seed", seed_summary), (best_name, best_summary)):
        flat = flatten_summary(summary)
        rows.append(
            [
                name,
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
    lines = [
        "# GEPA Reasoning V1 Run",
        "",
        f"- run_id: `{run_id}`",
        f"- budget: `{budget}`",
        f"- engine_mode: `{engine_mode}`",
        f"- reflection_lm: `{reflection_lm_name or 'off'}`",
        f"- status: `{status}`",
        f"- leakage_passed: `{leakage_report['passed']}`",
        f"- num_candidates: `{result_metrics.get('num_candidates', 'n/a')}`",
        f"- total_metric_calls: `{result_metrics.get('total_metric_calls', 'n/a')}`",
        "",
        "## Candidate Comparison",
        "",
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
        ),
        "",
        f"Best candidate: `{best_name}`",
        "",
        "## Seed Eval",
        "",
        render_report(seed_summary),
        "",
        "## Best Eval",
        "",
        render_report(best_summary),
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("gepa_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reflection_lm_name = resolve_reflection_lm_name(args.reflection_lm)

    adapter = ReasoningGEPAAdapter(seed=args.seed)
    leakage_report = run_leakage_red_team(adapter, dict(SEED_CANDIDATE), budget=min(24, args.budget), seed=args.seed)
    (run_dir / "leakage_red_team.json").write_text(json.dumps(leakage_report, indent=2, ensure_ascii=True) + "\n")

    full_batch = adapter.sample_batch(budget=args.budget, seed=args.seed)
    trainset, valset = split_train_val(full_batch)
    seed_eval = adapter.evaluate(batch=full_batch, candidate=dict(SEED_CANDIDATE), capture_traces=True)
    seed_summary = adapter.summary_for(seed_eval)
    reflective_dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), seed_eval, list(SEED_CANDIDATE.keys()))
    (run_dir / "seed_candidate.json").write_text(candidate_to_json(dict(SEED_CANDIDATE)) + "\n")
    (run_dir / "seed_reflective_dataset.json").write_text(json.dumps(reflective_dataset, indent=2, ensure_ascii=True) + "\n")

    if args.proposal_mode == "reflection_lm":
        if not reflection_lm_name:
            raise ValueError("--reflection-lm must be set when --proposal-mode=reflection_lm")
        engine_mode = f"reflection_lm:{reflection_lm_name}"
        custom_candidate_proposer = None
        reflection_lm = reflection_lm_name
    elif args.proposal_mode == "custom":
        engine_mode = "custom_proposer"
        custom_candidate_proposer = make_custom_candidate_proposer()
        reflection_lm = None
    else:
        if reflection_lm_name:
            engine_mode = f"reflection_lm:{reflection_lm_name}"
            custom_candidate_proposer = None
            reflection_lm = reflection_lm_name
        else:
            engine_mode = "custom_proposer"
            custom_candidate_proposer = make_custom_candidate_proposer()
            reflection_lm = None

    run_config = {
        "run_id": run_id,
        "budget": args.budget,
        "seed": args.seed,
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "proposal_mode": args.proposal_mode,
        "engine_mode": engine_mode,
        "reflection_lm": reflection_lm_name,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=True) + "\n")

    status = "benchmark_candidate"
    best_name = "seed"
    best_candidate = dict(SEED_CANDIDATE)
    result_dict = {"num_candidates": 0, "total_metric_calls": 0}

    if leakage_report["passed"]:
        result = optimize(
            seed_candidate=dict(SEED_CANDIDATE),
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            custom_candidate_proposer=custom_candidate_proposer,
            candidate_selection_strategy="pareto",
            frontier_type="objective",
            reflection_minibatch_size=min(args.reflection_minibatch_size, max(1, len(trainset))),
            max_metric_calls=args.max_metric_calls,
            run_dir=str(run_dir / "gepa_engine"),
            display_progress_bar=False,
            seed=args.seed,
            raise_on_exception=True,
            use_merge=False,
        )
        result_dict = result.to_dict()
        best_candidate = dict(result.best_candidate)
        best_name = "gepa_best"
        (run_dir / "gepa_result.json").write_text(json.dumps(result_dict, indent=2, ensure_ascii=True) + "\n")
    else:
        status = "benchmark_discard"

    best_eval = adapter.evaluate(batch=full_batch, candidate=best_candidate, capture_traces=True)
    best_summary = adapter.summary_for(best_eval)
    (run_dir / "best_candidate.json").write_text(candidate_to_json(best_candidate) + "\n")
    (run_dir / "best_summary.json").write_text(json.dumps(best_summary, indent=2, ensure_ascii=True) + "\n")

    if status != "benchmark_discard":
        status = "benchmark_win" if compare_candidates(flatten_summary(best_summary), flatten_summary(seed_summary)) > 0 else "benchmark_candidate"

    report = render_smoke_report(
        run_id,
        args.budget,
        engine_mode,
        reflection_lm_name,
        leakage_report,
        seed_summary,
        best_summary,
        best_name,
        status,
        result_dict,
    )
    (run_dir / "report.md").write_text(report + "\n")

    append_ledger_row(
        LEDGER_PATH,
        {
            "run_id": run_id,
            "status": status,
            "engine_mode": engine_mode,
            "reflection_lm": reflection_lm_name,
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
            "report_path": str((run_dir / "report.md").as_posix()),
        },
    )

    print(report)


if __name__ == "__main__":
    main()
