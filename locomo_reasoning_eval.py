from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from locomo_eval import (
    answer_matches,
    events_by_conversation,
    headers_by_conversation,
    is_abstain_like,
    load_jsonl,
    load_metadata,
    normalize_text,
)
from locomo_mode_adapter import (
    QUESTION_WORDS,
    build_event_candidates,
    build_header_candidates,
    content_terms,
    infer_hint_terms,
)
from reasoning_layer import infer_answer_style, run_reasoning
from reasoning_layer_schema import EvaluationBatch, MemoryContext, MemoryEvidence


OPEN_QUERY_MODES = ("current", "temporal", "multi_hop", "abstain_like")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GEPA v1 reasoning-layer LoCoMo evaluator.")
    parser.add_argument("--events", default="locomo_adapted/events.jsonl")
    parser.add_argument("--headers", default="locomo_adapted/headers.jsonl")
    parser.add_argument("--queries", default="locomo_adapted/queries.jsonl")
    parser.add_argument("--metadata", default="locomo_adapted/metadata.json")
    parser.add_argument("--candidate-json", default="")
    parser.add_argument("--budget", type=int, default=0, help="Optional evaluation budget. 0 means full dataset.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--write-markdown", default="")
    parser.add_argument("--write-json", default="")
    return parser.parse_args()


def build_profile(question: str, attribute: str | None) -> dict[str, Any]:
    subject_terms = {
        token.lower()
        for token in question.split()
        if token[:1].isupper() and token.rstrip("?:!.,").lower() not in QUESTION_WORDS
    }
    policy_mode = "temporal" if question.lower().startswith("when ") or " how long " in f" {question.lower()} " else "current"
    return {
        "question": question,
        "question_lower": question.lower(),
        "policy_mode": policy_mode,
        "query_terms": content_terms(question),
        "hint_terms": infer_hint_terms(question),
        "subject_terms": subject_terms,
        "attribute": attribute,
        "answer_style": infer_answer_style(question, policy_mode),
    }


def to_memory_context(query: dict, conversation_events: list[dict], conversation_headers: list[dict], top_k: int = 8) -> MemoryContext:
    profile = build_profile(query.get("text", ""), query.get("attribute"))
    event_candidates = build_event_candidates(query, conversation_events, profile)
    header_candidates = build_header_candidates(query, conversation_headers, conversation_events, profile)
    merged = sorted(event_candidates + header_candidates, key=lambda item: (item["score"], item["session_idx"]), reverse=True)[:top_k]
    evidence_items = [
        MemoryEvidence(
            kind=item["kind"],
            score=float(item["score"]),
            text=item["text"],
            value=item.get("value"),
            entity=item.get("entity"),
            attribute=item.get("attribute"),
            regime=item.get("regime"),
            session_idx=int(item["session_idx"]),
            session_time=item.get("session_time"),
            dia_ids=list(item.get("dia_ids", [])),
        )
        for item in merged
    ]
    return MemoryContext(
        query_id=query["query_id"],
        question=query.get("text", ""),
        entity=query.get("entity"),
        attribute=query.get("attribute"),
        regime=query.get("regime"),
        evidence_items=evidence_items,
    )


def evidence_overlap(support_items: list[MemoryEvidence], gold_dia_ids: list[str]) -> bool:
    if not gold_dia_ids:
        return False
    gold = set(gold_dia_ids)
    for item in support_items:
        if gold.intersection(item.dia_ids):
            return True
    return False


def explanation_quality(question: str, predicted_answer: str | None, explanation: str, support_items: list[MemoryEvidence], should_abstain: bool) -> str:
    if not explanation.strip():
        return "weak"
    explanation_terms = content_terms(explanation)
    if should_abstain and not any(term in explanation.lower() for term in ("abstain", "ground", "conflict", "evidence", "support")):
        return "weak"
    if not support_items:
        return "weak"
    support_terms = content_terms(" ".join(item.text for item in support_items[:2]))
    if support_terms and not explanation_terms.intersection(support_terms):
        return "unsupported"
    if predicted_answer and normalize_text(predicted_answer) not in normalize_text(explanation):
        answer_terms = content_terms(predicted_answer)
        if answer_terms and not explanation_terms.intersection(answer_terms):
            return "weak"
    return "good"


def make_group() -> dict[str, int]:
    return {
        "count": 0,
        "joint_correct": 0,
        "predicted_abstain": 0,
        "gold_abstain": 0,
        "answer_match": 0,
        "evidence_hit": 0,
    }


def pct(value: int, total: int) -> float:
    return 0.0 if total <= 0 else value / total


def summarize_group(group: dict[str, int]) -> dict[str, float | int]:
    return {
        "count": group["count"],
        "joint_acc": pct(group["joint_correct"], group["count"]),
        "pred_abstain_rate": pct(group["predicted_abstain"], group["count"]),
        "gold_abstain_rate": pct(group["gold_abstain"], group["count"]),
        "answer_match_rate": pct(group["answer_match"], group["count"]),
        "evidence_hit_rate": pct(group["evidence_hit"], group["count"]),
    }


def update_group(group: dict[str, int], joint_correct: int, pred_abs: int, gold_abs: int, answer_match: int, evidence_hit: int):
    group["count"] += 1
    group["joint_correct"] += joint_correct
    group["predicted_abstain"] += pred_abs
    group["gold_abstain"] += gold_abs
    group["answer_match"] += answer_match
    group["evidence_hit"] += evidence_hit


def aggregate_objectives(examples: list[dict[str, Any]]) -> dict[str, float]:
    answerable = [row for row in examples if row["gold_abstain"] == 0]
    abstain_cases = [row for row in examples if row["gold_abstain"] == 1]
    return {
        "answerable_accuracy": pct(sum(row["joint_correct"] for row in answerable), len(answerable)),
        "abstain_precision": pct(sum(1 for row in examples if row["pred_abs"] == 1 and row["gold_abstain"] == 1), sum(row["pred_abs"] for row in examples)),
        "abstain_recall": pct(sum(1 for row in abstain_cases if row["pred_abs"] == 1), len(abstain_cases)),
        "false_abstain_penalty": pct(sum(row["false_abstain"] for row in answerable), len(answerable)),
        "false_confident_answer_penalty": pct(sum(row["false_confident_answer"] for row in examples), len(examples)),
    }


def scalar_score(row: dict[str, Any]) -> float:
    return (
        1.0 * row["joint_correct"]
        - 0.75 * row["false_abstain"]
        - 0.35 * row["false_confident_answer"]
    )


def offline_slice_name(query: dict) -> str:
    mode = query.get("query_mode")
    if mode in {"current", "temporal", "multi_hop"}:
        return mode
    if is_abstain_like(query.get("gold_answer", "")):
        return "abstain_like"
    return mode or "unknown"


def sample_queries(queries: list[dict], budget: int, seed: int) -> list[dict]:
    if budget <= 0 or budget >= len(queries):
        return list(queries)
    grouped = defaultdict(list)
    for query in queries:
        grouped[offline_slice_name(query)].append(query)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)
    ordered_modes = ["current", "temporal", "multi_hop", "abstain_like", "historical", "adversarial", "unknown"]
    batch = []
    while len(batch) < budget:
        added = False
        for mode in ordered_modes:
            rows = grouped.get(mode)
            if rows:
                batch.append(rows.pop())
                added = True
                if len(batch) >= budget:
                    break
        if not added:
            break
    return batch


def evaluate_reasoning_dataset(
    candidate: dict[str, str],
    events_path: str,
    headers_path: str,
    queries_path: str,
    metadata_path: str | None = None,
    budget: int = 0,
    seed: int = 13,
    capture_traces: bool = False,
    max_failure_examples: int = 20,
    max_explanation_audit_examples: int = 12,
):
    events = load_jsonl(Path(events_path))
    headers = load_jsonl(Path(headers_path))
    queries = load_jsonl(Path(queries_path))
    metadata = load_metadata(Path(metadata_path)) if metadata_path else {}
    query_batch = sample_queries(queries, budget, seed)
    conversations = events_by_conversation(events)
    headers_by_conv = headers_by_conversation(headers)
    return evaluate_reasoning_batch(
        candidate,
        query_batch,
        conversations,
        headers_by_conv,
        metadata=metadata,
        capture_traces=capture_traces,
        max_failure_examples=max_failure_examples,
        max_explanation_audit_examples=max_explanation_audit_examples,
    )


def evaluate_reasoning_batch(
    candidate: dict[str, str],
    query_batch: list[dict],
    conversations: dict[str, list[dict]],
    headers_by_conv: dict[str, list[dict]],
    metadata: dict[str, Any] | None = None,
    capture_traces: bool = False,
    max_failure_examples: int = 20,
    max_explanation_audit_examples: int = 12,
):
    _ = metadata or {}
    outputs = []
    scores = []
    trajectories = []
    objective_scores = []
    overall = make_group()
    by_slice = defaultdict(make_group)
    by_predicted_mode = defaultdict(make_group)
    failure_buckets = Counter(
        {
            "temporal_selection_error": 0,
            "missing_evidence": 0,
            "false_abstain": 0,
            "multi_hop_failure": 0,
            "attribute_mismatch": 0,
            "entity_mismatch": 0,
            "false_confident_answer": 0,
            "explanation_quality": 0,
        }
    )
    failure_examples = []
    explanation_audit = []
    rows = []

    for query in query_batch:
        context = to_memory_context(query, conversations.get(query["conversation_id"], []), headers_by_conv.get(query["conversation_id"], []))
        reasoning = run_reasoning(candidate, context)
        pred_abs = int(reasoning.should_abstain)
        gold_abs = int(is_abstain_like(query.get("gold_answer", "")))
        evidence_hit = int(evidence_overlap(reasoning.support_items, query.get("gold_evidence_dia_ids") or []))
        answer_match = 0 if pred_abs else int(answer_matches(reasoning.answer_candidate, query.get("gold_answer", "")))
        joint_correct = int(pred_abs == 1) if gold_abs else int(pred_abs == 0 and answer_match == 1)
        false_abstain = int(gold_abs == 0 and pred_abs == 1)
        false_confident_answer = int(pred_abs == 0 and joint_correct == 0)
        explanation_grade = explanation_quality(
            query.get("text", ""),
            reasoning.answer_candidate,
            reasoning.explanation,
            reasoning.support_items,
            reasoning.should_abstain,
        )

        update_group(overall, joint_correct, pred_abs, gold_abs, answer_match, evidence_hit)
        update_group(by_slice[offline_slice_name(query)], joint_correct, pred_abs, gold_abs, answer_match, evidence_hit)
        update_group(by_predicted_mode[reasoning.query_mode], joint_correct, pred_abs, gold_abs, answer_match, evidence_hit)

        support = reasoning.support_items[0] if reasoning.support_items else None
        if support is not None and support.attribute != query.get("attribute"):
            failure_buckets["attribute_mismatch"] += 1
        if support is not None and support.entity != query.get("entity"):
            failure_buckets["entity_mismatch"] += 1
        if explanation_grade != "good":
            failure_buckets["explanation_quality"] += 1

        bucket_name = None
        if false_abstain:
            bucket_name = "false_abstain"
        elif false_confident_answer:
            failure_buckets["false_confident_answer"] += 1
            if not context.evidence_items:
                bucket_name = "missing_evidence"
            elif reasoning.query_mode == "multi_hop":
                bucket_name = "multi_hop_failure"
            elif reasoning.query_mode in {"current", "temporal"} and not evidence_hit:
                bucket_name = "temporal_selection_error"
            else:
                bucket_name = "false_confident_answer"
        if bucket_name:
            failure_buckets[bucket_name] += 1

        row = {
            "query_id": query["query_id"],
            "query_mode": query.get("query_mode", "unknown"),
            "predicted_mode": reasoning.query_mode,
            "joint_correct": joint_correct,
            "pred_abs": pred_abs,
            "gold_abstain": gold_abs,
            "answer_match": answer_match,
            "evidence_hit": evidence_hit,
            "false_abstain": false_abstain,
            "false_confident_answer": false_confident_answer,
            "explanation_quality": explanation_grade,
        }
        rows.append(row)
        objectives = {
            "answerable_accuracy": float(joint_correct if gold_abs == 0 else 0),
            "abstain_precision": float(1 if pred_abs == 1 and gold_abs == 1 else 0),
            "abstain_recall": float(1 if pred_abs == 1 and gold_abs == 1 else 0),
            "false_abstain_penalty": float(false_abstain),
            "false_confident_answer_penalty": float(false_confident_answer),
        }
        objective_scores.append(objectives)
        scores.append(scalar_score({"joint_correct": joint_correct, "false_abstain": false_abstain, "false_confident_answer": false_confident_answer}))

        output = reasoning.to_dict()
        outputs.append(output)
        trajectory = {
            "query_id": query["query_id"],
            "component_hint": reasoning.query_mode,
            "Inputs": {
                "question": query.get("text", ""),
                "entity": query.get("entity"),
                "attribute": query.get("attribute"),
                "regime": query.get("regime"),
                "retrieved_evidence_summary": [item.to_dict() for item in reasoning.support_items[:3]],
            },
            "Generated Outputs": {
                "query_mode": reasoning.query_mode,
                "answer": reasoning.answer_candidate,
                "abstain": reasoning.should_abstain,
                "explanation": reasoning.explanation,
                "confidence_band": reasoning.confidence_band,
            },
            "Feedback": {
                "gold_mode_label": query.get("query_mode"),
                "gold_answer": query.get("gold_answer"),
                "gold_abstain": bool(gold_abs),
                "failure_bucket": bucket_name,
                "explanation_quality": explanation_grade,
                "reflection_target": reasoning.query_mode,
                "answer_match": bool(answer_match),
                "evidence_hit": bool(evidence_hit),
            },
        }
        trajectories.append(trajectory)

        if bucket_name and len(failure_examples) < max_failure_examples:
            failure_examples.append(
                {
                    "query_id": query["query_id"],
                    "bucket": bucket_name,
                    "question": query.get("text", ""),
                    "gold_answer": query.get("gold_answer", ""),
                    "predicted_answer": reasoning.answer_candidate,
                    "predicted_mode": reasoning.query_mode,
                    "explanation": reasoning.explanation,
                    "support_text": support.text if support else "",
                    "support_dia_ids": support.dia_ids if support else [],
                }
            )
        if len(explanation_audit) < max_explanation_audit_examples and (explanation_grade != "good" or pred_abs == 1):
            explanation_audit.append(
                {
                    "query_id": query["query_id"],
                    "question": query.get("text", ""),
                    "predicted_mode": reasoning.query_mode,
                    "predicted_answer": reasoning.answer_candidate,
                    "should_abstain": reasoning.should_abstain,
                    "explanation": reasoning.explanation,
                    "support_text": support.text if support else "",
                    "gold_answer": query.get("gold_answer", ""),
                    "gold_abstain": bool(gold_abs),
                    "explanation_quality": explanation_grade,
                }
            )

    objectives_summary = aggregate_objectives(rows)
    slice_summary = {name: summarize_group(group) for name, group in sorted(by_slice.items())}
    predicted_mode_summary = {name: summarize_group(group) for name, group in sorted(by_predicted_mode.items())}
    aggregate_metrics = {
        "query_count": overall["count"],
        "joint_answer_or_abstain_acc": pct(overall["joint_correct"], overall["count"]),
        "abstain_precision": objectives_summary["abstain_precision"],
        "abstain_recall": objectives_summary["abstain_recall"],
        "answer_match_rate": pct(overall["answer_match"], overall["count"]),
        "answer_evidence_recall": pct(overall["evidence_hit"], overall["count"]),
        "current_joint_acc": slice_summary.get("current", {}).get("joint_acc", 0.0),
        "temporal_joint_acc": slice_summary.get("temporal", {}).get("joint_acc", 0.0),
        "multi_hop_joint_acc": slice_summary.get("multi_hop", {}).get("joint_acc", 0.0),
        "abstain_like_acc": slice_summary.get("abstain_like", {}).get("joint_acc", 0.0),
        "false_abstain_penalty": objectives_summary["false_abstain_penalty"],
        "false_confident_answer_penalty": objectives_summary["false_confident_answer_penalty"],
    }
    summary = {
        "aggregate_metrics": aggregate_metrics,
        "objective_summary": objectives_summary,
        "by_slice": slice_summary,
        "by_predicted_mode": predicted_mode_summary,
        "failure_buckets": dict(failure_buckets),
        "failure_examples": failure_examples,
        "explanation_audit": explanation_audit,
    }
    batch = EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
        objective_scores=objective_scores,
        aggregate_metrics=summary,
    )
    if capture_traces:
        return batch, summary
    return batch, summary


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


def render_report(summary: dict[str, Any]) -> str:
    headline = summary["aggregate_metrics"]
    lines = [
        "# LoCoMo Reasoning Eval",
        "",
        f"- query_count: `{headline['query_count']}`",
        f"- joint_answer_or_abstain_acc: `{headline['joint_answer_or_abstain_acc']:.4f}`",
        f"- abstain_precision: `{headline['abstain_precision']:.4f}`",
        f"- abstain_recall: `{headline['abstain_recall']:.4f}`",
        f"- answer_match_rate: `{headline['answer_match_rate']:.4f}`",
        f"- answer_evidence_recall: `{headline['answer_evidence_recall']:.4f}`",
        "",
        "## Objectives",
        "",
    ]
    objective_rows = [[name, f"{value:.4f}"] for name, value in summary["objective_summary"].items()]
    lines.append(markdown_table(["objective", "value"], objective_rows))
    lines.extend(["", "## Offline Slices", ""])
    slice_rows = []
    for name, stats in summary["by_slice"].items():
        slice_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["slice", "count", "joint_acc", "pred_abstain", "answer_match", "evidence_hit"], slice_rows))
    lines.extend(["", "## Predicted Modes", ""])
    pred_rows = []
    for name, stats in summary["by_predicted_mode"].items():
        pred_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["predicted_mode", "count", "joint_acc", "pred_abstain", "answer_match", "evidence_hit"], pred_rows))
    lines.extend(["", "## Failure Buckets", ""])
    failure_rows = [[name, count] for name, count in sorted(summary["failure_buckets"].items())]
    lines.append(markdown_table(["bucket", "count"], failure_rows))
    if summary["failure_examples"]:
        lines.extend(["", "## Failure Examples", ""])
        for idx, item in enumerate(summary["failure_examples"], start=1):
            lines.append(
                f"{idx}. `{item['query_id']}` | bucket=`{item['bucket']}` | mode=`{item['predicted_mode']}` | "
                f"gold=`{item['gold_answer']}` | pred=`{item['predicted_answer'] or 'ABSTAIN'}`"
            )
            lines.append(f"   question: {item['question']}")
            if item["support_text"]:
                lines.append(f"   support_text: {item['support_text']}")
    if summary["explanation_audit"]:
        lines.extend(["", "## Explanation Audit", ""])
        for idx, item in enumerate(summary["explanation_audit"], start=1):
            lines.append(
                f"{idx}. `{item['query_id']}` | quality=`{item['explanation_quality']}` | "
                f"abstain=`{item['should_abstain']}` | pred=`{item['predicted_answer'] or 'ABSTAIN'}`"
            )
            lines.append(f"   question: {item['question']}")
            lines.append(f"   explanation: {item['explanation']}")
            if item["support_text"]:
                lines.append(f"   support_text: {item['support_text']}")
    lines.append("")
    return "\n".join(lines)


def main():
    from reasoning_layer_prompts import SEED_CANDIDATE

    args = parse_args()
    candidate = dict(SEED_CANDIDATE)
    if args.candidate_json:
        candidate = json.loads(Path(args.candidate_json).read_text())
    _, summary = evaluate_reasoning_dataset(
        candidate=candidate,
        events_path=args.events,
        headers_path=args.headers,
        queries_path=args.queries,
        metadata_path=args.metadata,
        budget=args.budget,
        seed=args.seed,
        capture_traces=True,
    )
    report = render_report(summary)
    if args.write_markdown:
        path = Path(args.write_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n")
    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
