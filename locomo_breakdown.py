import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


OPEN_ANSWER_MARKERS = (
    "not mentioned",
    "no information",
    "unknown",
    "cannot be determined",
    "not enough information",
    "not provided",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the adapted LoCoMo benchmark slices without touching the UCMD baseline.")
    parser.add_argument("--queries", default="locomo_adapted/queries.jsonl", help="Path to the adapted LoCoMo queries JSONL file.")
    parser.add_argument("--metadata", default="locomo_adapted/metadata.json", help="Path to the adapted LoCoMo metadata JSON file.")
    parser.add_argument("--write-markdown", default="", help="Optional path for a generated Markdown snapshot report.")
    parser.add_argument("--write-json", default="", help="Optional path for a generated JSON summary.")
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


def is_abstain_like(answer: str):
    lowered = (answer or "").strip().lower()
    return lowered == "" or any(marker in lowered for marker in OPEN_ANSWER_MARKERS)


def pct(count: int, total: int):
    if total <= 0:
        return 0.0
    return 100.0 * count / total


def summarize_bucket(rows):
    total = len(rows)
    abstain_like = sum(is_abstain_like(row.get("gold_answer", "")) for row in rows)
    answerable = total - abstain_like
    with_evidence = sum(bool(row.get("gold_evidence_dia_ids")) for row in rows)
    no_evidence = total - with_evidence
    multi_evidence = sum(len(row.get("gold_evidence_dia_ids") or []) > 1 for row in rows)
    total_evidence = sum(len(row.get("gold_evidence_dia_ids") or []) for row in rows)
    return {
        "count": total,
        "share_pct": round(pct(total, 1), 4),
        "answerable": answerable,
        "abstain_like": abstain_like,
        "with_evidence": with_evidence,
        "no_evidence": no_evidence,
        "multi_evidence": multi_evidence,
        "avg_evidence_ids": round(total_evidence / max(1, total), 4),
    }


def summarize_named_groups(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    total = len(rows)
    out = {}
    for key in sorted(groups, key=lambda value: str(value)):
        summary = summarize_bucket(groups[key])
        summary["share_pct"] = round(pct(summary["count"], total), 2)
        out[str(key)] = summary
    return out


def build_cross_tab(rows):
    modes = sorted({row.get("query_mode", "unknown") for row in rows})
    categories = sorted({str(row.get("category", "unknown")) for row in rows}, key=lambda value: (value == "unknown", value))
    counts = {mode: Counter() for mode in modes}
    for row in rows:
        counts[row.get("query_mode", "unknown")][str(row.get("category", "unknown"))] += 1
    return {"modes": modes, "categories": categories, "counts": {mode: {category: counts[mode][category] for category in categories} for mode in modes}}


def top_attributes_by_mode(rows, top_k: int = 5):
    per_mode = defaultdict(Counter)
    for row in rows:
        per_mode[row.get("query_mode", "unknown")][row.get("attribute", "unknown")] += 1
    out = {}
    for mode in sorted(per_mode):
        out[mode] = per_mode[mode].most_common(top_k)
    return out


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


def render_markdown(summary):
    metadata = summary["metadata"]
    overall = summary["overall"]
    lines = [
        "# LoCoMo Breakdown",
        "",
        f"- source: `{metadata.get('source_path', 'unknown')}`",
        f"- adapter_version: `{metadata.get('adapter_version', 'unknown')}`",
        f"- locomo_version: `{metadata.get('locomo_version', 'unknown')}`",
        f"- queries: `{overall['count']}`",
        f"- answerable: `{overall['answerable']}`",
        f"- abstain_like: `{overall['abstain_like']}`",
        f"- with_evidence: `{overall['with_evidence']}`",
        f"- avg_evidence_ids: `{overall['avg_evidence_ids']}`",
        "",
        "## Query Mode",
        "",
    ]

    mode_rows = []
    for mode, stats in summary["by_query_mode"].items():
        mode_rows.append(
            [
                mode,
                stats["count"],
                f"{stats['share_pct']:.2f}",
                stats["answerable"],
                stats["abstain_like"],
                stats["with_evidence"],
                stats["no_evidence"],
                stats["multi_evidence"],
                f"{stats['avg_evidence_ids']:.2f}",
            ]
        )
    lines.append(
        markdown_table(
            ["query_mode", "count", "share_%", "answerable", "abstain_like", "with_evidence", "no_evidence", "multi_evidence", "avg_evidence_ids"],
            mode_rows,
        )
    )
    lines.extend(["", "## Category", ""])

    category_rows = []
    for category, stats in summary["by_category"].items():
        category_rows.append(
            [
                category,
                stats["count"],
                f"{stats['share_pct']:.2f}",
                stats["answerable"],
                stats["abstain_like"],
                stats["with_evidence"],
                stats["no_evidence"],
                stats["multi_evidence"],
                f"{stats['avg_evidence_ids']:.2f}",
            ]
        )
    lines.append(
        markdown_table(
            ["category", "count", "share_%", "answerable", "abstain_like", "with_evidence", "no_evidence", "multi_evidence", "avg_evidence_ids"],
            category_rows,
        )
    )
    lines.extend(["", "## Query Mode x Category", ""])

    cross_tab = summary["query_mode_by_category"]
    headers = ["query_mode", *cross_tab["categories"]]
    cross_rows = []
    for mode in cross_tab["modes"]:
        row = [mode]
        row.extend(cross_tab["counts"][mode][category] for category in cross_tab["categories"])
        cross_rows.append(row)
    lines.append(markdown_table(headers, cross_rows))
    lines.extend(["", "## Top Attributes Per Query Mode", ""])

    for mode, pairs in summary["top_attributes_by_query_mode"].items():
        rendered = ", ".join(f"`{name}` ({count})" for name, count in pairs) or "none"
        lines.append(f"- `{mode}`: {rendered}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    queries_path = Path(args.queries)
    metadata_path = Path(args.metadata)
    queries = load_jsonl(queries_path)
    metadata = load_metadata(metadata_path)

    overall = summarize_bucket(queries)
    overall["share_pct"] = 100.0
    summary = {
        "metadata": metadata,
        "overall": overall,
        "by_query_mode": summarize_named_groups(queries, lambda row: row.get("query_mode", "unknown")),
        "by_category": summarize_named_groups(queries, lambda row: row.get("category", "unknown")),
        "query_mode_by_category": build_cross_tab(queries),
        "top_attributes_by_query_mode": top_attributes_by_mode(queries),
    }

    report = render_markdown(summary)
    print(report)

    if args.write_markdown:
        markdown_path = Path(args.write_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report + "\n")

    if args.write_json:
        json_path = Path(args.write_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
