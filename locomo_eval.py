import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from locomo_mode_adapter import predict_from_locomo


OPEN_ANSWER_MARKERS = (
    "not mentioned",
    "no information",
    "unknown",
    "cannot be determined",
    "not enough information",
    "not provided",
)

CONFLICT_THRESHOLD = 0.35


def parse_args():
    parser = argparse.ArgumentParser(description="Run the UCMD V3.6 LoCoMo eval-only benchmark over adapted events and queries.")
    parser.add_argument("--events", default="locomo_adapted/events.jsonl", help="Path to adapted LoCoMo events.")
    parser.add_argument("--headers", default="locomo_adapted/headers.jsonl", help="Path to adapted LoCoMo headers.")
    parser.add_argument("--queries", default="locomo_adapted/queries.jsonl", help="Path to adapted LoCoMo queries.")
    parser.add_argument("--metadata", default="locomo_adapted/metadata.json", help="Path to adapted LoCoMo metadata.")
    parser.add_argument("--write-markdown", default="", help="Optional Markdown report output path.")
    parser.add_argument("--write-json", default="", help="Optional JSON report output path.")
    parser.add_argument("--max-failures", type=int, default=20, help="Maximum number of failure examples to keep in the report.")
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_metadata(path: Path):
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def normalize_text(text: str):
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def is_abstain_like(answer: str):
    lowered = (answer or "").strip().lower()
    return lowered == "" or any(marker in lowered for marker in OPEN_ANSWER_MARKERS)


def answer_matches(predicted: str | None, gold: str):
    if not predicted:
        return False
    pred_norm = normalize_text(predicted)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if len(pred_norm) >= 4 and pred_norm in gold_norm:
        return True
    if len(gold_norm) >= 4 and gold_norm in pred_norm:
        return True
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if pred_tokens and gold_tokens:
        overlap = len(pred_tokens & gold_tokens)
        if overlap >= max(2, min(len(pred_tokens), len(gold_tokens))):
            return True
    return False


def events_by_conversation(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["conversation_id"]].append(event)
    for conversation_id in grouped:
        grouped[conversation_id].sort(key=lambda item: (item["timestamp"], item["event_id"]))
    return grouped


def headers_by_conversation(headers):
    grouped = defaultdict(list)
    for header in headers:
        grouped[header["conversation_id"]].append(header)
    for conversation_id in grouped:
        grouped[conversation_id].sort(key=lambda item: (item["session_idx"], item["header_id"]))
    return grouped


def value_stats(events: list[dict]):
    masses = Counter(event["value"] for event in events)
    if not masses:
        return {
            "top_value": None,
            "top_mass": 0.0,
            "top2_mass": 0.0,
            "gap": 0.0,
            "conflict_mass": 0.0,
            "total_mass": 0.0,
            "masses": {},
        }
    ordered = masses.most_common()
    top_value, top_mass = ordered[0]
    top2_mass = ordered[1][1] if len(ordered) > 1 else 0.0
    total_mass = sum(masses.values())
    return {
        "top_value": top_value,
        "top_mass": float(top_mass),
        "top2_mass": float(top2_mass),
        "gap": float(top_mass - top2_mass),
        "conflict_mass": float(top2_mass / max(1.0, total_mass)),
        "total_mass": float(total_mass),
        "masses": dict(masses),
    }


def choose_prediction(query: dict, scoped_events: list[dict], candidates: list[dict]):
    stats = value_stats(candidates)
    if not candidates or not stats["top_value"]:
        return {
            "abstain": True,
            "predicted_value": None,
            "stats": stats,
            "support_events": [],
        }

    mode = query.get("query_mode", "current")
    if mode == "current":
        unique_values = {event["value"] for event in candidates}
        abstain = len(unique_values) != 1
    else:
        abstain = stats["gap"] <= 0.0 or stats["conflict_mass"] > CONFLICT_THRESHOLD

    support_events = [event for event in candidates if event["value"] == stats["top_value"]]
    if not support_events and scoped_events:
        support_events = list(scoped_events)
    return {
        "abstain": abstain,
        "predicted_value": None if abstain else stats["top_value"],
        "stats": stats,
        "support_events": support_events,
    }


def evidence_overlap(events: list[dict], gold_evidence_ids: list[str]):
    if not gold_evidence_ids:
        return False
    gold = set(gold_evidence_ids)
    for event in events:
        if gold.intersection(event.get("dia_ids", [])):
            return True
    return False


def pct(count: int, total: int):
    if total <= 0:
        return 0.0
    return count / total


def make_group():
    return {
        "count": 0,
        "joint_correct": 0,
        "predicted_abstain": 0,
        "gold_abstain": 0,
        "answer_match": 0,
        "evidence_hit": 0,
    }


def summarize_group(group: dict):
    return {
        "count": group["count"],
        "joint_acc": pct(group["joint_correct"], group["count"]),
        "pred_abstain_rate": pct(group["predicted_abstain"], group["count"]),
        "gold_abstain_rate": pct(group["gold_abstain"], group["count"]),
        "answer_match_rate": pct(group["answer_match"], group["count"]),
        "evidence_hit_rate": pct(group["evidence_hit"], group["count"]),
    }


def markdown_table(headers, rows):
    widths = [len(header) for header in headers]
    string_rows = [[str(cell) for cell in row] for row in rows]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row):
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [fmt(headers), divider]
    lines.extend(fmt(row) for row in string_rows)
    return "\n".join(lines)


def render_report(summary: dict):
    lines = [
        "# LoCoMo Eval",
        "",
        f"- query_count: `{summary['headline']['query_count']}`",
        f"- joint_answer_or_abstain_acc: `{summary['headline']['joint_answer_or_abstain_acc']:.4f}`",
        f"- abstain_precision: `{summary['headline']['abstain_precision']:.4f}`",
        f"- abstain_recall: `{summary['headline']['abstain_recall']:.4f}`",
        f"- answer_match_rate: `{summary['headline']['answer_match_rate']:.4f}`",
        f"- answer_evidence_recall: `{summary['headline']['answer_evidence_recall']:.4f}`",
        "",
        "## Query Mode",
        "",
    ]

    query_mode_rows = []
    for name, stats in summary["by_query_mode"].items():
        query_mode_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['gold_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["query_mode", "count", "joint_acc", "pred_abstain", "gold_abstain", "answer_match", "evidence_hit"], query_mode_rows))
    lines.extend(["", "## Policy Mode", ""])

    policy_mode_rows = []
    for name, stats in summary["by_policy_mode"].items():
        policy_mode_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['gold_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["policy_mode", "count", "joint_acc", "pred_abstain", "gold_abstain", "answer_match", "evidence_hit"], policy_mode_rows))
    lines.extend(["", "## Query To Policy", ""])

    policy_map_rows = []
    for name, stats in summary["query_to_policy_mode"].items():
        policy_map_rows.append([name, stats["count"], stats["dominant_policy_mode"], f"{stats['dominant_policy_share']:.4f}"])
    lines.append(markdown_table(["query_mode", "count", "dominant_policy_mode", "dominant_policy_share"], policy_map_rows))
    lines.extend(["", "## Category", ""])

    category_rows = []
    for name, stats in summary["by_category"].items():
        category_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['gold_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["category", "count", "joint_acc", "pred_abstain", "gold_abstain", "answer_match", "evidence_hit"], category_rows))
    lines.extend(["", "## Focus Slices", ""])

    focus_rows = []
    for name, stats in summary["focus_groups"].items():
        focus_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["slice", "count", "joint_acc", "pred_abstain", "answer_match", "evidence_hit"], focus_rows))
    lines.extend(["", "## Policy Focus", ""])

    policy_focus_rows = []
    for name, stats in summary["policy_focus_groups"].items():
        policy_focus_rows.append(
            [
                name,
                stats["count"],
                f"{stats['joint_acc']:.4f}",
                f"{stats['pred_abstain_rate']:.4f}",
                f"{stats['answer_match_rate']:.4f}",
                f"{stats['evidence_hit_rate']:.4f}",
            ]
        )
    lines.append(markdown_table(["policy_slice", "count", "joint_acc", "pred_abstain", "answer_match", "evidence_hit"], policy_focus_rows))
    lines.extend(["", "## Failure Buckets", ""])

    failure_rows = [[name, count] for name, count in sorted(summary["failure_buckets"].items())]
    lines.append(markdown_table(["bucket", "count"], failure_rows))
    if summary["failure_examples"]:
        lines.extend(["", "## Failure Examples", ""])
        for idx, item in enumerate(summary["failure_examples"], start=1):
            lines.append(
                f"{idx}. `{item['query_id']}` | mode=`{item['query_mode']}` | policy=`{item['policy_mode']}` | bucket=`{item['bucket']}` | "
                f"gold=`{item['gold_answer']}` | pred=`{item['predicted_answer'] or 'ABSTAIN'}`"
            )
            lines.append(f"   question: {item['question']}")
            if item["support_value"]:
                lines.append(f"   support_value: {item['support_value']}")
            if item["support_text"]:
                lines.append(f"   support_text: {item['support_text']}")
            if item["support_dia_ids"]:
                lines.append(f"   support_dia_ids: {item['support_dia_ids']}")
    lines.append("")
    return "\n".join(lines)


def evaluate_locomo(
    events_path: str,
    queries_path: str,
    metadata_path: str | None = None,
    headers_path: str = "",
    write_markdown: str = "",
    write_json: str = "",
    max_failures: int = 20,
):
    events = load_jsonl(Path(events_path))
    headers_file = Path(headers_path) if headers_path else Path(events_path).with_name("headers.jsonl")
    headers = load_jsonl(headers_file) if headers_file.exists() else []
    queries = load_jsonl(Path(queries_path))
    metadata = load_metadata(Path(metadata_path)) if metadata_path else {}
    conversations = events_by_conversation(events)
    headers_by_conv = headers_by_conversation(headers)

    overall = make_group()
    by_query_mode = defaultdict(make_group)
    by_policy_mode = defaultdict(make_group)
    by_category = defaultdict(make_group)
    focus_groups = {
        "current": make_group(),
        "temporal": make_group(),
        "multi_hop": make_group(),
        "abstain_like": make_group(),
    }
    policy_focus_groups = {
        "current": make_group(),
        "temporal": make_group(),
        "multi_hop": make_group(),
        "abstain_like": make_group(),
    }
    query_to_policy_mode = defaultdict(Counter)
    failure_buckets = Counter(
        {
            "temporal_selection_error": 0,
            "missing_evidence": 0,
            "false_abstain": 0,
            "multi_hop_failure": 0,
            "attribute_mismatch": 0,
            "entity_mismatch": 0,
        }
    )
    failure_examples = []

    abstain_tp = 0
    abstain_pred = 0
    abstain_gold = 0

    for query in queries:
        conversation_events = conversations.get(query["conversation_id"], [])
        conversation_headers = headers_by_conv.get(query["conversation_id"], [])
        profile, choice = predict_from_locomo(query, conversation_events, conversation_headers)
        policy_mode = choice.get("policy_mode") or profile.get("policy_mode", "current")
        pred_abs = int(choice["abstain"])
        gold_abs = int(is_abstain_like(query.get("gold_answer", "")))
        pred_answer = choice["predicted_value"]
        evidence_hit = evidence_overlap(choice["support_events"], query.get("gold_evidence_dia_ids") or [])
        answer_match = 0 if pred_abs else int(answer_matches(pred_answer, query.get("gold_answer", "")))
        joint_correct = int(pred_abs == 1) if gold_abs else int(pred_abs == 0 and answer_match == 1)

        overall["count"] += 1
        overall["joint_correct"] += joint_correct
        overall["predicted_abstain"] += pred_abs
        overall["gold_abstain"] += gold_abs
        overall["answer_match"] += answer_match
        overall["evidence_hit"] += int(evidence_hit)

        mode_bucket = by_query_mode[query.get("query_mode", "unknown")]
        mode_bucket["count"] += 1
        mode_bucket["joint_correct"] += joint_correct
        mode_bucket["predicted_abstain"] += pred_abs
        mode_bucket["gold_abstain"] += gold_abs
        mode_bucket["answer_match"] += answer_match
        mode_bucket["evidence_hit"] += int(evidence_hit)

        policy_bucket = by_policy_mode[policy_mode]
        policy_bucket["count"] += 1
        policy_bucket["joint_correct"] += joint_correct
        policy_bucket["predicted_abstain"] += pred_abs
        policy_bucket["gold_abstain"] += gold_abs
        policy_bucket["answer_match"] += answer_match
        policy_bucket["evidence_hit"] += int(evidence_hit)

        category_bucket = by_category[str(query.get("category", "unknown"))]
        category_bucket["count"] += 1
        category_bucket["joint_correct"] += joint_correct
        category_bucket["predicted_abstain"] += pred_abs
        category_bucket["gold_abstain"] += gold_abs
        category_bucket["answer_match"] += answer_match
        category_bucket["evidence_hit"] += int(evidence_hit)

        query_to_policy_mode[query.get("query_mode", "unknown")][policy_mode] += 1

        if query.get("query_mode") in focus_groups:
            focus = focus_groups[query["query_mode"]]
            focus["count"] += 1
            focus["joint_correct"] += joint_correct
            focus["predicted_abstain"] += pred_abs
            focus["gold_abstain"] += gold_abs
            focus["answer_match"] += answer_match
            focus["evidence_hit"] += int(evidence_hit)
        if gold_abs:
            focus = focus_groups["abstain_like"]
            focus["count"] += 1
            focus["joint_correct"] += joint_correct
            focus["predicted_abstain"] += pred_abs
            focus["gold_abstain"] += gold_abs
            focus["answer_match"] += answer_match
            focus["evidence_hit"] += int(evidence_hit)

        if policy_mode in policy_focus_groups:
            focus = policy_focus_groups[policy_mode]
            focus["count"] += 1
            focus["joint_correct"] += joint_correct
            focus["predicted_abstain"] += pred_abs
            focus["gold_abstain"] += gold_abs
            focus["answer_match"] += answer_match
            focus["evidence_hit"] += int(evidence_hit)
        if gold_abs:
            focus = policy_focus_groups["abstain_like"]
            focus["count"] += 1
            focus["joint_correct"] += joint_correct
            focus["predicted_abstain"] += pred_abs
            focus["gold_abstain"] += gold_abs
            focus["answer_match"] += answer_match
            focus["evidence_hit"] += int(evidence_hit)

        abstain_tp += int(pred_abs == 1 and gold_abs == 1)
        abstain_pred += pred_abs
        abstain_gold += gold_abs

        failure_tags = []
        bucket_name = None
        support = choice["support_events"][0] if choice["support_events"] else None
        if support is not None and support.get("attribute") != query.get("attribute"):
            failure_buckets["attribute_mismatch"] += 1
            failure_tags.append("attribute_mismatch")
        if support is not None and support.get("entity") != query.get("entity"):
            failure_buckets["entity_mismatch"] += 1
            failure_tags.append("entity_mismatch")
        if gold_abs == 0 and pred_abs == 1:
            bucket_name = "false_abstain"
            failure_buckets[bucket_name] += 1
        elif gold_abs == 0 and joint_correct == 0:
            if not choice["support_events"]:
                bucket_name = "missing_evidence"
            elif policy_mode == "multi_hop" or query.get("query_mode") == "multi_hop":
                bucket_name = "multi_hop_failure"
            elif policy_mode in {"current", "temporal"} and not evidence_hit:
                bucket_name = "temporal_selection_error"
            elif not evidence_hit:
                bucket_name = "missing_evidence"
            if bucket_name:
                failure_buckets[bucket_name] += 1

        if bucket_name and len(failure_examples) < max_failures:
            failure_examples.append(
                {
                    "query_id": query["query_id"],
                    "query_mode": query.get("query_mode", "unknown"),
                    "policy_mode": policy_mode,
                    "bucket": bucket_name,
                    "question": query.get("text", ""),
                    "gold_answer": query.get("gold_answer", ""),
                    "predicted_answer": pred_answer,
                    "support_value": support.get("value") if support else None,
                    "support_text": support.get("text") if support else None,
                    "support_dia_ids": support.get("dia_ids") if support else [],
                    "failure_tags": failure_tags,
                }
            )

    by_query_mode_summary = {name: summarize_group(bucket) for name, bucket in sorted(by_query_mode.items())}
    by_policy_mode_summary = {name: summarize_group(bucket) for name, bucket in sorted(by_policy_mode.items())}
    by_category_summary = {name: summarize_group(bucket) for name, bucket in sorted(by_category.items())}
    focus_summary = {name: summarize_group(bucket) for name, bucket in focus_groups.items()}
    policy_focus_summary = {name: summarize_group(bucket) for name, bucket in policy_focus_groups.items()}
    query_to_policy_summary = {}
    for query_mode, counts in sorted(query_to_policy_mode.items()):
        total = sum(counts.values())
        dominant_policy_mode, dominant_count = counts.most_common(1)[0]
        query_to_policy_summary[query_mode] = {
            "count": total,
            "dominant_policy_mode": dominant_policy_mode,
            "dominant_policy_share": pct(dominant_count, total),
            "counts": dict(counts),
        }
    headline = {
        "query_count": overall["count"],
        "joint_answer_or_abstain_acc": pct(overall["joint_correct"], overall["count"]),
        "abstain_precision": pct(abstain_tp, abstain_pred),
        "abstain_recall": pct(abstain_tp, abstain_gold),
        "answer_match_rate": pct(overall["answer_match"], overall["count"]),
        "answer_evidence_recall": pct(overall["evidence_hit"], overall["count"]),
        "current_joint_acc": focus_summary["current"]["joint_acc"],
        "temporal_joint_acc": focus_summary["temporal"]["joint_acc"],
        "multi_hop_joint_acc": focus_summary["multi_hop"]["joint_acc"],
        "abstain_like_acc": focus_summary["abstain_like"]["joint_acc"],
    }
    summary = {
        "metadata": metadata,
        "headline": headline,
        "by_query_mode": by_query_mode_summary,
        "by_policy_mode": by_policy_mode_summary,
        "query_to_policy_mode": query_to_policy_summary,
        "by_category": by_category_summary,
        "focus_groups": focus_summary,
        "policy_focus_groups": policy_focus_summary,
        "failure_buckets": dict(failure_buckets),
        "failure_examples": failure_examples,
    }

    report = render_report(summary)
    if write_markdown:
        markdown_path = Path(write_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report + "\n")
    if write_json:
        json_path = Path(write_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    return summary, report


def main():
    args = parse_args()
    summary, report = evaluate_locomo(
        args.events,
        args.queries,
        args.metadata,
        headers_path=args.headers,
        write_markdown=args.write_markdown,
        write_json=args.write_json,
        max_failures=args.max_failures,
    )
    _ = summary
    print(report)


if __name__ == "__main__":
    main()
