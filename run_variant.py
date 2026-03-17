import argparse
import csv
import functools
import subprocess
import time
from pathlib import Path

import train


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = REPO_ROOT / "results.tsv"
DECISION_TOLERANCE = 0.001


def run_git(args):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def head_commit() -> str:
    return run_git(["rev-parse", "HEAD"]).stdout.strip()


def changed_paths_for_run():
    status = run_git(["status", "--porcelain=v1", "--untracked-files=all"]).stdout.splitlines()
    paths = []
    for line in status:
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path == "results.tsv" or path.startswith("__pycache__/") or path.endswith(".pyc"):
            continue
        paths.append(path)
    return sorted(set(paths))


def git_diff_summary(paths):
    if not paths:
        return ""
    return run_git(["diff", "--stat", "--", *paths]).stdout.strip()


def stage_and_commit(paths, run_id: str):
    if not paths:
        raise RuntimeError("No code or doc changes found for this run. results.tsv alone is not enough.")
    run_git(["add", "-A", "--", *paths])
    run_git(["commit", "-m", f"run:{run_id}", "--only", "--", *paths])


def read_results_rows(path: Path):
    train.ensure_results_header(str(path))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def metric(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def compare_runs(candidate: dict, incumbent: dict, tolerance: float) -> int:
    keys = [
        ("hard_joint_answer_or_abstain_acc", True),
        ("hard_latest_slice_recall", True),
        ("hard_latest_slice_conflict_accuracy", True),
        ("joint_answer_or_abstain_acc", True),
        ("hard_avg_retrieval_latency_ms", False),
    ]
    for key, higher_is_better in keys:
        cand = metric(candidate, key)
        base = metric(incumbent, key)
        if higher_is_better:
            if cand > base + tolerance:
                return 1
            if cand < base - tolerance:
                return -1
        else:
            if cand < base - tolerance:
                return 1
            if cand > base + tolerance:
                return -1
    return 0


def best_keep_row(rows, tolerance: float):
    keep_rows = [row for row in rows if row.get("status") == "keep"]
    if not keep_rows:
        return None

    best = keep_rows[0]
    for row in keep_rows[1:]:
        if compare_runs(row, best, tolerance) > 0:
            best = row
    return best


def build_row(run_id: str, git_commit: str, git_parent_commit: str, description: str, status: str, reason: str, summary: dict):
    final_metrics = summary["final_metrics"]
    hard_metrics = summary.get("hard_metrics") or {}
    config = summary["config"]
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "git_parent_commit": git_parent_commit,
        "description": description,
        "status": status,
        "decision_reason": reason,
        "samples": config["samples"],
        "epochs": config["epochs"],
        "joint_answer_or_abstain_acc": final_metrics["joint_answer_or_abstain_acc"],
        "abstain_accuracy": final_metrics["abstain_accuracy"],
        "abstain_precision": final_metrics["abstain_precision"],
        "abstain_recall": final_metrics["abstain_recall"],
        "raw_recall_at_k": final_metrics["raw_recall_at_k"],
        "latest_slice_recall": final_metrics["latest_slice_recall"],
        "latest_slice_conflict_accuracy": final_metrics["latest_slice_conflict_accuracy"],
        "regime_mismatch": final_metrics["regime_mismatch"],
        "time_bucket_mismatch": final_metrics["time_bucket_mismatch"],
        "attribute_mismatch": final_metrics["attribute_mismatch"],
        "wrong_answer_should_abstain": final_metrics["wrong_answer_should_abstain"],
        "wrong_answer_should_answer": final_metrics["wrong_answer_should_answer"],
        "false_abstain": final_metrics["false_abstain"],
        "false_merge_rate": final_metrics["false_merge_rate"],
        "unnecessary_branch_rate": final_metrics["unnecessary_branch_rate"],
        "header_fake_fact_rate": final_metrics["header_fake_fact_rate"],
        "avg_header_value_purity": final_metrics["avg_header_value_purity"],
        "avg_headers_used": final_metrics["avg_headers_used"],
        "avg_retrieval_latency_ms": final_metrics["avg_retrieval_latency_ms"],
        "hard_joint_answer_or_abstain_acc": hard_metrics.get("joint_answer_or_abstain_acc"),
        "hard_abstain_accuracy": hard_metrics.get("abstain_accuracy"),
        "hard_abstain_precision": hard_metrics.get("abstain_precision"),
        "hard_abstain_recall": hard_metrics.get("abstain_recall"),
        "hard_raw_recall_at_k": hard_metrics.get("raw_recall_at_k"),
        "hard_latest_slice_recall": hard_metrics.get("latest_slice_recall"),
        "hard_latest_slice_conflict_accuracy": hard_metrics.get("latest_slice_conflict_accuracy"),
        "hard_regime_mismatch": hard_metrics.get("regime_mismatch"),
        "hard_time_bucket_mismatch": hard_metrics.get("time_bucket_mismatch"),
        "hard_attribute_mismatch": hard_metrics.get("attribute_mismatch"),
        "hard_wrong_answer_should_abstain": hard_metrics.get("wrong_answer_should_abstain"),
        "hard_wrong_answer_should_answer": hard_metrics.get("wrong_answer_should_answer"),
        "hard_false_abstain": hard_metrics.get("false_abstain"),
        "hard_false_merge_rate": hard_metrics.get("false_merge_rate"),
        "hard_unnecessary_branch_rate": hard_metrics.get("unnecessary_branch_rate"),
        "hard_header_fake_fact_rate": hard_metrics.get("header_fake_fact_rate"),
        "hard_avg_header_value_purity": hard_metrics.get("avg_header_value_purity"),
        "hard_avg_headers_used": hard_metrics.get("avg_headers_used"),
        "hard_avg_retrieval_latency_ms": hard_metrics.get("avg_retrieval_latency_ms"),
    }


def decide_status(candidate: dict, incumbent: dict | None, tolerance: float):
    if metric(candidate, "regime_mismatch") > 0 or metric(candidate, "hard_regime_mismatch") > 0:
        return "discard", "regime_mismatch > 0 reopened a solved failure mode"

    if incumbent is None:
        return "keep", "no prior keep row in the locked ledger; establishing the first reference candidate"

    incumbent_run = incumbent.get("run_id", "unknown")
    if metric(candidate, "hard_joint_answer_or_abstain_acc") < metric(incumbent, "hard_joint_answer_or_abstain_acc") - tolerance:
        return "discard", f"hard joint accuracy fell below keep run {incumbent_run}"
    if metric(candidate, "hard_latest_slice_conflict_accuracy") < metric(incumbent, "hard_latest_slice_conflict_accuracy") - tolerance:
        return "discard", f"hard latest-slice conflict accuracy fell below keep run {incumbent_run}"
    if metric(candidate, "hard_joint_answer_or_abstain_acc") > metric(incumbent, "hard_joint_answer_or_abstain_acc") + tolerance:
        return "keep", f"hard joint accuracy beat keep run {incumbent_run}"
    if metric(candidate, "hard_latest_slice_recall") > metric(incumbent, "hard_latest_slice_recall") + tolerance:
        return "keep", f"hard latest-slice recall beat keep run {incumbent_run} at tied hard joint accuracy"
    if metric(candidate, "hard_latest_slice_recall") < metric(incumbent, "hard_latest_slice_recall") - tolerance:
        return "discard", f"hard latest-slice recall fell below keep run {incumbent_run}"
    if metric(candidate, "hard_latest_slice_conflict_accuracy") > metric(incumbent, "hard_latest_slice_conflict_accuracy") + tolerance:
        return "keep", f"hard latest-slice conflict accuracy beat keep run {incumbent_run}"
    if metric(candidate, "joint_answer_or_abstain_acc") > metric(incumbent, "joint_answer_or_abstain_acc") + tolerance:
        return "keep", f"standard joint accuracy beat keep run {incumbent_run} after hard-eval tie"
    if metric(candidate, "joint_answer_or_abstain_acc") < metric(incumbent, "joint_answer_or_abstain_acc") - tolerance:
        return "discard", f"standard joint accuracy fell below keep run {incumbent_run} after hard-eval tie"
    if metric(candidate, "hard_avg_retrieval_latency_ms") < metric(incumbent, "hard_avg_retrieval_latency_ms") - tolerance:
        return "keep", f"hard retrieval latency improved against keep run {incumbent_run}"
    return "discard", f"ties best keep run {incumbent_run} within tolerance; incumbent remains the reference"


def parse_args():
    parser = argparse.ArgumentParser(description="Commit, run, evaluate, and log one UCMD variant.")
    parser.add_argument("--description", required=True, help="Precise human-readable summary of the single hypothesis in this run.")
    parser.add_argument("--tolerance", type=float, default=DECISION_TOLERANCE, help="Tie tolerance for metric comparisons.")
    return parser.parse_args()


def main():
    args = parse_args()
    description = train.tsv_safe(args.description)
    if not description:
        raise RuntimeError("Description must not be empty.")

    paths = changed_paths_for_run()
    print("changed files for this run:")
    for path in paths:
        print(f"  {path}")
    diff_summary = git_diff_summary(paths)
    if diff_summary:
        print("\ndiff summary:")
        print(diff_summary)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    git_parent_commit = head_commit()
    stage_and_commit(paths, run_id)
    git_commit = head_commit()
    print(f"\ncommitted run:{run_id}")
    print(f"git_parent_commit: {git_parent_commit}")
    print(f"git_commit:        {git_commit}")

    summary = train.run_experiment()
    if summary.get("non_finite"):
        status = "discard"
        reason = "non-finite loss or gradient halted the run early"
        row = build_row(run_id, git_commit, git_parent_commit, description, status, reason, summary)
        train.append_results_row(str(RESULTS_PATH), row)
        print(f"\nledger decision: {status}")
        print(f"decision_reason: {reason}")
        return

    rows = read_results_rows(RESULTS_PATH)
    incumbent = best_keep_row(rows, args.tolerance)
    candidate = build_row(run_id, git_commit, git_parent_commit, description, "", "", summary)
    status, reason = decide_status(candidate, incumbent, args.tolerance)
    candidate["status"] = status
    candidate["decision_reason"] = reason
    train.append_results_row(str(RESULTS_PATH), candidate)

    print(f"\nledger decision: {status}")
    print(f"decision_reason: {reason}")


if __name__ == "__main__":
    main()
