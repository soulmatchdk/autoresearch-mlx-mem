from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from gepa import optimize

from gepa_reasoning_adapter import ReasoningGEPAAdapter
from leakage_red_team import run_leakage_red_team
from locomo_reasoning_eval import offline_slice_name, render_report
from reasoning_layer_prompts import SEED_CANDIDATE, SMOKE_RUN_CONFIG, candidate_to_json, clone_candidate
from reasoning_layer_schema import (
    ABSTAIN_PROFILES,
    ANSWER_STYLE_POLICIES,
    CURRENT_STRATEGIES,
    GENERIC_ANSWER_RULES,
    MODE_ROUTERS,
    MULTI_HOP_STRATEGIES,
    TEMPORAL_STRATEGIES,
)


LEDGER_PATH = Path("gepa_runs/ledger.tsv")
DECISION_TOLERANCE = 0.001
REFLECTION_LM_ENV = "GEPA_REFLECTION_LM"
LEDGER_HEADERS = [
    "run_id",
    "status",
    "track",
    "engine_mode",
    "reflection_lm",
    "budget",
    "holdout_budget",
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
    "seed_holdout_joint_acc",
    "best_holdout_joint_acc",
    "seed_holdout_answerable_accuracy",
    "best_holdout_answerable_accuracy",
    "best_candidate_name",
    "report_path",
]

TRACK_COMPONENTS = {
    "all": list(SEED_CANDIDATE.keys()),
    "mode_abstain": [
        "mode_router",
        "abstain_profile",
        "generic_answer_rule",
    ],
    "temporal_selection": [
        "temporal_strategy",
        "answer_style",
    ],
}

TRACK_SLICES = {
    "all": ["current", "temporal", "multi_hop", "abstain_like"],
    "mode_abstain": ["current", "temporal", "multi_hop", "abstain_like"],
    "temporal_selection": ["temporal"],
}

TRACK_OBJECTIVES = {
    "all": [
        "joint_reward",
        "answerable_reward",
        "abstain_precision_reward",
        "abstain_recall_reward",
        "avoid_false_abstain",
        "avoid_false_confident_answer",
        "temporal_joint_reward",
        "temporal_evidence_reward",
        "answer_evidence_reward",
    ],
    "mode_abstain": [
        "answerable_reward",
        "abstain_precision_reward",
        "abstain_recall_reward",
        "avoid_false_abstain",
        "avoid_false_confident_answer",
    ],
    "temporal_selection": [
        "temporal_joint_reward",
        "temporal_evidence_reward",
        "answer_evidence_reward",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GEPA v1 reasoning-layer slice with the official GEPA engine.")
    parser.add_argument("--track", choices=tuple(TRACK_COMPONENTS.keys()), default="all")
    parser.add_argument("--budget", type=int, default=SMOKE_RUN_CONFIG["eval_budget"])
    parser.add_argument("--holdout-budget", type=int, default=32)
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


def compare_candidates_for_track(track: str, left: dict[str, float], right: dict[str, float], tolerance: float = DECISION_TOLERANCE) -> int:
    if track == "mode_abstain":
        keys = [
            ("answerable_accuracy", True),
            ("abstain_precision", True),
            ("abstain_recall", True),
            ("false_abstain_penalty", False),
            ("false_confident_answer_penalty", False),
        ]
    elif track == "temporal_selection":
        keys = [
            ("temporal_joint_acc", True),
            ("answer_evidence_recall", True),
            ("false_abstain_penalty", False),
            ("false_confident_answer_penalty", False),
        ]
    else:
        return compare_candidates(left, right, tolerance)
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


def track_components(track: str) -> list[str]:
    return list(TRACK_COMPONENTS[track])


def track_slices(track: str) -> list[str]:
    return list(TRACK_SLICES[track])


def track_objectives(track: str) -> list[str]:
    return list(TRACK_OBJECTIVES[track])


def balanced_track_batch(queries: list[dict[str, Any]], track: str, budget: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in track_slices(track)}
    fallback: list[dict[str, Any]] = []
    for query in queries:
        slice_name = offline_slice_name(query)
        if slice_name in grouped:
            grouped[slice_name].append(query)
        elif track == "all":
            fallback.append(query)
    if budget <= 0:
        batch: list[dict[str, Any]] = []
        for slice_name in track_slices(track):
            batch.extend(grouped.get(slice_name, []))
        batch.extend(fallback)
        return batch
    local_rng = random.Random(seed)
    for rows in grouped.values():
        local_rng.shuffle(rows)
    local_rng.shuffle(fallback)
    batch: list[dict[str, Any]] = []
    ordered = track_slices(track)
    while len(batch) < budget:
        added = False
        for slice_name in ordered:
            rows = grouped.get(slice_name, [])
            if rows:
                batch.append(rows.pop())
                added = True
                if len(batch) >= budget:
                    break
        if not added:
            break
    while len(batch) < budget and fallback:
        batch.append(fallback.pop())
    return batch


def split_train_val(batch: list[dict[str, Any]], track: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(batch) < 12:
        return batch, batch
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in track_slices(track)}
    extra: list[dict[str, Any]] = []
    for query in batch:
        slice_name = offline_slice_name(query)
        if slice_name in grouped:
            grouped[slice_name].append(query)
        else:
            extra.append(query)
    trainset: list[dict[str, Any]] = []
    valset: list[dict[str, Any]] = []
    for rows in grouped.values():
        if not rows:
            continue
        val_size = max(1, len(rows) // 3)
        if val_size >= len(rows) and len(rows) > 1:
            val_size = len(rows) - 1
        valset.extend(rows[:val_size])
        trainset.extend(rows[val_size:])
    if extra:
        split = max(1, len(extra) // 3)
        valset.extend(extra[:split])
        trainset.extend(extra[split:])
    if not trainset:
        trainset = list(valset)
    if not valset:
        valset = list(trainset)
    return trainset, valset


def make_holdout_batch(all_queries: list[dict[str, Any]], used_batch: list[dict[str, Any]], track: str, budget: int, seed: int) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    used_ids = {query["query_id"] for query in used_batch}
    remaining = [query for query in all_queries if query["query_id"] not in used_ids]
    if not remaining:
        remaining = list(all_queries)
    return balanced_track_batch(remaining, track=track, budget=budget, seed=seed)


COMPONENT_OPTIONS = {
    "mode_router": list(MODE_ROUTERS),
    "current_strategy": list(CURRENT_STRATEGIES),
    "temporal_strategy": list(TEMPORAL_STRATEGIES),
    "multi_hop_strategy": list(MULTI_HOP_STRATEGIES),
    "answer_style": list(ANSWER_STYLE_POLICIES),
    "abstain_profile": list(ABSTAIN_PROFILES),
    "generic_answer_rule": list(GENERIC_ANSWER_RULES),
}


def normalize_choice(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_")


def cycle_component_choice(candidate: dict[str, str], component: str, step: int = 1) -> dict[str, str]:
    tuned = clone_candidate(candidate)
    options = COMPONENT_OPTIONS[component]
    current = normalize_choice(candidate.get(component, ""))
    try:
        idx = options.index(current)
    except ValueError:
        idx = 0
    tuned[component] = options[(idx + step) % len(options)]
    return tuned


def apply_component_tune(candidate: dict[str, str], component: str, iteration: int) -> dict[str, str]:
    if component not in COMPONENT_OPTIONS:
        return clone_candidate(candidate)
    return cycle_component_choice(candidate, component, step=max(1, iteration % len(COMPONENT_OPTIONS[component])))


def reflective_signals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_buckets = Counter()
    gold_modes = Counter()
    missed_types = Counter()
    bad_patterns = Counter()
    fix_hints = Counter()
    answer_matches = 0
    evidence_hits = 0
    for row in rows:
        feedback = row.get("Feedback", {})
        failure_bucket = feedback.get("failure_bucket")
        if failure_bucket:
            failure_buckets[failure_bucket] += 1
        gold_mode = feedback.get("gold_mode_label")
        if gold_mode:
            gold_modes[gold_mode] += 1
        missed = feedback.get("missed_evidence_type")
        if missed:
            missed_types[missed] += 1
        pattern = feedback.get("bad_decision_pattern")
        if pattern:
            bad_patterns[pattern] += 1
        hint = feedback.get("minimal_fix_hint")
        if hint:
            fix_hints[hint] += 1
        if feedback.get("answer_match"):
            answer_matches += 1
        if feedback.get("evidence_hit"):
            evidence_hits += 1
    return {
        "count": len(rows),
        "failure_buckets": failure_buckets,
        "gold_modes": gold_modes,
        "missed_types": missed_types,
        "bad_patterns": bad_patterns,
        "fix_hints": fix_hints,
        "answer_match_rate": answer_matches / max(1, len(rows)),
        "evidence_hit_rate": evidence_hits / max(1, len(rows)),
    }


def apply_reflective_tune(candidate: dict[str, str], component: str, rows: list[dict[str, Any]], iteration: int) -> dict[str, str]:
    tuned = clone_candidate(candidate)
    signals = reflective_signals(rows)
    failures = signals["failure_buckets"]
    gold_modes = signals["gold_modes"]
    missed_types = signals["missed_types"]
    if component == "mode_router":
        if gold_modes.get("temporal", 0) >= max(gold_modes.get("current", 0), gold_modes.get("multi_hop", 0)) and (
            missed_types.get("time_anchor_alignment", 0) > 0 or failures.get("temporal_selection_error", 0) > 0
        ):
            tuned["mode_router"] = "temporal_first"
        elif gold_modes.get("multi_hop", 0) > gold_modes.get("current", 0):
            tuned["mode_router"] = "multi_hop_sensitive"
        elif gold_modes.get("abstain_like", 0) > 0 and failures.get("false_confident_answer", 0) > failures.get("false_abstain", 0):
            tuned["mode_router"] = "abstain_sensitive"
        else:
            tuned["mode_router"] = "balanced_temporal"
    elif component == "current_strategy":
        tuned["current_strategy"] = "latest_with_time_anchor" if missed_types.get("time_anchor_alignment", 0) > 0 else "latest_only"
    elif component == "temporal_strategy":
        if missed_types.get("session_time_grounding", 0) > 0 or missed_types.get("temporal_grounding", 0) > 0:
            tuned["temporal_strategy"] = "explicit_or_session_time"
        elif missed_types.get("time_anchor_alignment", 0) > 0:
            tuned["temporal_strategy"] = "latest_with_time_anchor"
        else:
            tuned["temporal_strategy"] = "ordered_history"
    elif component == "multi_hop_strategy":
        if missed_types.get("multi_evidence_synthesis", 0) > 0 or failures.get("multi_hop_failure", 0) > 0:
            tuned["multi_hop_strategy"] = "aggregate_three_hops"
        else:
            tuned["multi_hop_strategy"] = "aggregate_two_hops"
    elif component == "abstain_profile":
        if failures.get("false_abstain", 0) > failures.get("false_confident_answer", 0):
            tuned["abstain_profile"] = "answerable_friendly"
        elif failures.get("false_confident_answer", 0) > 0:
            tuned["abstain_profile"] = "strict"
        else:
            tuned["abstain_profile"] = "balanced"
    elif component == "generic_answer_rule":
        if failures.get("false_confident_answer", 0) > 0 or missed_types.get("specific_answer_span", 0) > 0:
            tuned["generic_answer_rule"] = "reject_full_sentence"
        else:
            tuned["generic_answer_rule"] = "reject_long_span"
    elif component == "answer_style":
        if gold_modes.get("temporal", 0) > 0:
            tuned["answer_style"] = "short_date"
        elif gold_modes.get("multi_hop", 0) > 0:
            tuned["answer_style"] = "short_slot_value"
        elif failures.get("false_confident_answer", 0) > 0:
            tuned["answer_style"] = "short_entity"
        else:
            tuned["answer_style"] = "auto"
    else:
        tuned = apply_component_tune(tuned, component, iteration)
    return tuned


def make_custom_candidate_proposer(allowed_components: list[str] | None = None):
    state = {"calls": 0}
    allowed = set(allowed_components or [])

    def proposer(candidate: dict[str, str], reflective_dataset: dict[str, list[dict[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        state["calls"] += 1
        targets = list(components_to_update or [])
        if not targets:
            targets = ["mode_router"]
        if allowed:
            targets = [component for component in targets if component in allowed]
        if not targets and allowed:
            targets = [sorted(allowed)[0]]
        tuned = clone_candidate(candidate)
        for component in targets:
            rows = reflective_dataset.get(component, [])
            if rows:
                tuned = apply_reflective_tune(tuned, component, rows, state["calls"])
            elif component in tuned:
                tuned = apply_component_tune(tuned, component, state["calls"])
        return tuned

    return proposer


class FixedTrackModuleSelector:
    def __init__(self, components: list[str]):
        self.components = list(components)
        self.idx = 0

    def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate) -> list[str]:
        _ = (state, trajectories, subsample_scores, candidate_idx)
        available = [component for component in self.components if component in candidate]
        if not available:
            return list(candidate.keys())
        choice = available[self.idx % len(available)]
        self.idx += 1
        return [choice]


def component_option_text(component: str) -> str:
    options = COMPONENT_OPTIONS.get(component, [])
    return ", ".join(f"`{option}`" for option in options)


def build_reflection_prompt_templates(track: str, components: list[str]) -> dict[str, str]:
    common = (
        "You are improving one discrete policy component of a reasoning system.\n\n"
        "Current component text:\n```\n<curr_param>\n```\n\n"
        "Here are reflective examples with Inputs, Generated Outputs, and Feedback:\n```\n<side_info>\n```\n\n"
        "Choose one allowed value for this component. Make a small policy change, not a rewrite. "
        "Use the feedback fields such as missed_evidence_type, bad_decision_pattern, should_have_done, and minimal_fix_hint "
        "to repair one concrete failure mode.\n"
    )
    prompts: dict[str, str] = {}
    if track == "mode_abstain":
        delta = (
            "Goal: improve answerable questions and abstain balance.\n"
            "Do not worsen abstain precision on abstain-like questions.\n"
            "Focus only on routing to the right mode for answerable questions, letting compact grounded spans beat abstain, "
            "or rejecting generic copied answers.\n"
            "Return only one allowed option for this component within ``` blocks."
        )
        for component in components:
            prompts[component] = common + f"\nAllowed options for `{component}`: {component_option_text(component)}.\n\n" + delta
    elif track == "temporal_selection":
        delta = (
            "Goal: improve temporal answer selection without increasing false abstain.\n"
            "Focus only on indirect date grounding from session time, time-anchor alignment, and compact temporal answer synthesis.\n"
            "Do not change current or multi-hop behavior, and do not make generic abstain rules stricter.\n"
            "Return only one allowed option for this component within ``` blocks."
        )
        for component in components:
            prompts[component] = common + f"\nAllowed options for `{component}`: {component_option_text(component)}.\n\n" + delta
    return prompts


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
    track: str,
    budget: int,
    holdout_budget: int,
    components: list[str],
    engine_mode: str,
    reflection_lm_name: str,
    leakage_report: dict[str, Any],
    seed_summary: dict[str, Any],
    best_summary: dict[str, Any],
    seed_holdout_summary: dict[str, Any] | None,
    best_holdout_summary: dict[str, Any] | None,
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
        f"- track: `{track}`",
        f"- budget: `{budget}`",
        f"- holdout_budget: `{holdout_budget}`",
        f"- active_components: `{', '.join(components)}`",
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
    if seed_holdout_summary and best_holdout_summary:
        holdout_rows = []
        for name, summary in (("seed_holdout", seed_holdout_summary), (f"{best_name}_holdout", best_holdout_summary)):
            flat = flatten_summary(summary)
            holdout_rows.append(
                [
                    name,
                    f"{flat['current_joint_acc']:.4f}",
                    f"{flat['temporal_joint_acc']:.4f}",
                    f"{flat['multi_hop_joint_acc']:.4f}",
                    f"{flat['answerable_accuracy']:.4f}",
                    f"{flat['abstain_precision']:.4f}",
                    f"{flat['abstain_recall']:.4f}",
                ]
            )
        lines.extend(
            [
                "## Holdout Comparison",
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
                    ],
                    holdout_rows,
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("gepa_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reflection_lm_name = resolve_reflection_lm_name(args.reflection_lm)
    components = track_components(args.track)
    objectives = track_objectives(args.track)
    adapter = ReasoningGEPAAdapter(seed=args.seed, active_components=components, active_objectives=objectives)
    leakage_budget = min(24, args.budget) if args.budget > 0 else 24
    leakage_report = run_leakage_red_team(adapter, dict(SEED_CANDIDATE), budget=leakage_budget, seed=args.seed)
    (run_dir / "leakage_red_team.json").write_text(json.dumps(leakage_report, indent=2, ensure_ascii=True) + "\n")

    full_batch = balanced_track_batch(adapter.queries, track=args.track, budget=args.budget, seed=args.seed)
    holdout_batch = make_holdout_batch(adapter.queries, used_batch=full_batch, track=args.track, budget=args.holdout_budget, seed=args.seed + 101)
    trainset, valset = split_train_val(full_batch, track=args.track)
    seed_eval = adapter.evaluate(batch=full_batch, candidate=dict(SEED_CANDIDATE), capture_traces=True)
    seed_summary = adapter.summary_for(seed_eval)
    seed_holdout_summary = None
    if holdout_batch:
        seed_holdout_eval = adapter.evaluate(batch=holdout_batch, candidate=dict(SEED_CANDIDATE), capture_traces=False)
        seed_holdout_summary = adapter.summary_for(seed_holdout_eval)
        (run_dir / "seed_holdout_summary.json").write_text(json.dumps(seed_holdout_summary, indent=2, ensure_ascii=True) + "\n")
    reflective_dataset = adapter.make_reflective_dataset(dict(SEED_CANDIDATE), seed_eval, components)
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
        custom_candidate_proposer = make_custom_candidate_proposer(components)
        reflection_lm = None
    else:
        if reflection_lm_name:
            engine_mode = f"reflection_lm:{reflection_lm_name}"
            custom_candidate_proposer = None
            reflection_lm = reflection_lm_name
        else:
            engine_mode = "custom_proposer"
            custom_candidate_proposer = make_custom_candidate_proposer(components)
            reflection_lm = None

    run_config = {
        "run_id": run_id,
        "track": args.track,
        "active_components": components,
        "active_objectives": objectives,
        "budget": args.budget,
        "holdout_budget": args.holdout_budget,
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
    reflection_prompt_template = build_reflection_prompt_templates(args.track, components) if reflection_lm else None

    if leakage_report["passed"]:
        result = optimize(
            seed_candidate=dict(SEED_CANDIDATE),
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            reflection_prompt_template=reflection_prompt_template,
            custom_candidate_proposer=custom_candidate_proposer,
            candidate_selection_strategy="pareto",
            frontier_type="objective",
            reflection_minibatch_size=min(args.reflection_minibatch_size, max(1, len(trainset))),
            max_metric_calls=args.max_metric_calls,
            module_selector=FixedTrackModuleSelector(components),
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
    best_holdout_summary = None
    if holdout_batch:
        best_holdout_eval = adapter.evaluate(batch=holdout_batch, candidate=best_candidate, capture_traces=False)
        best_holdout_summary = adapter.summary_for(best_holdout_eval)
        (run_dir / "best_holdout_summary.json").write_text(json.dumps(best_holdout_summary, indent=2, ensure_ascii=True) + "\n")
    (run_dir / "best_candidate.json").write_text(candidate_to_json(best_candidate) + "\n")
    (run_dir / "best_summary.json").write_text(json.dumps(best_summary, indent=2, ensure_ascii=True) + "\n")

    if status != "benchmark_discard":
        comparison_seed = flatten_summary(seed_holdout_summary or seed_summary)
        comparison_best = flatten_summary(best_holdout_summary or best_summary)
        status = "benchmark_win" if compare_candidates_for_track(args.track, comparison_best, comparison_seed) > 0 else "benchmark_candidate"

    report = render_smoke_report(
        run_id,
        args.track,
        args.budget,
        args.holdout_budget,
        components,
        engine_mode,
        reflection_lm_name,
        leakage_report,
        seed_summary,
        best_summary,
        seed_holdout_summary,
        best_holdout_summary,
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
            "track": args.track,
            "engine_mode": engine_mode,
            "reflection_lm": reflection_lm_name,
            "budget": args.budget,
            "holdout_budget": args.holdout_budget,
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
            "seed_holdout_joint_acc": f"{flatten_summary(seed_holdout_summary or seed_summary)['joint_answer_or_abstain_acc']:.6f}",
            "best_holdout_joint_acc": f"{flatten_summary(best_holdout_summary or best_summary)['joint_answer_or_abstain_acc']:.6f}",
            "seed_holdout_answerable_accuracy": f"{flatten_summary(seed_holdout_summary or seed_summary)['answerable_accuracy']:.6f}",
            "best_holdout_answerable_accuracy": f"{flatten_summary(best_holdout_summary or best_summary)['answerable_accuracy']:.6f}",
            "best_candidate_name": best_name,
            "report_path": str((run_dir / "report.md").as_posix()),
        },
    )

    print(report)


if __name__ == "__main__":
    main()
