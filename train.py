"""
UCMD V3.6 baseline for the autoresearch-mlx repo.

This keeps the repo in the same Apple Silicon / MLX / agent-loop spirit,
with a single-file UCMD memory baseline centered on freshness-first answer selection.

What this file does:
- builds a small synthetic dataset in memory (or loads/dumps it outside the repo)
- trains only small MLX-native MLP modules and heads
- exposes a real Memory API for agent use
- reports UCMD metrics for the wrapper loop and direct inspection
"""

import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np


SEED = 42
random.seed(SEED)
mx.random.seed(SEED)


@dataclass
class Config:
    dataset_path: str = os.getenv("UCMD_DATA", "")
    results_path: str = str(Path(__file__).with_name("results.tsv"))

    n_samples: int = int(os.getenv("UCMD_SAMPLES", 1000))
    hard_eval_samples: int = int(os.getenv("UCMD_HARD_EVAL_SAMPLES", 180))
    min_events: int = int(os.getenv("UCMD_MIN_EVENTS", 10))
    max_events: int = int(os.getenv("UCMD_MAX_EVENTS", 14))

    d_field: int = int(os.getenv("UCMD_D_FIELD", 16))
    d_model: int = int(os.getenv("UCMD_D_MODEL", 64))
    d_hidden: int = int(os.getenv("UCMD_D_HIDDEN", 128))

    h_max: int = int(os.getenv("UCMD_H_MAX", 12))
    n_hot: int = int(os.getenv("UCMD_N_HOT", 384))
    k_scope: int = int(os.getenv("UCMD_K_SCOPE", 32))
    k_rerank: int = int(os.getenv("UCMD_K_RERANK", 8))

    epochs: int = int(os.getenv("UCMD_EPOCHS", 8))
    batch_size: int = int(os.getenv("UCMD_BATCH_SIZE", 16))
    lr: float = float(os.getenv("UCMD_LR", 3e-4))
    wd: float = float(os.getenv("UCMD_WD", 1e-4))
    grad_clip_norm: float = float(os.getenv("UCMD_GRAD_CLIP", 1.0))

    lambda_answer: float = 1.0
    lambda_abstain: float = 0.5
    lambda_branch: float = 0.3
    lambda_scope: float = 0.2
    lambda_retrieval: float = 0.3

    semantic_beta: float = float(os.getenv("UCMD_SEM_BETA", 0.01))
    quality_tau: float = float(os.getenv("UCMD_QUALITY_TAU", 0.60))
    abstain_tau: float = float(os.getenv("UCMD_ABSTAIN_TAU", 0.65))
    abstain_evidence_floor: float = float(os.getenv("UCMD_ABS_EVIDENCE_FLOOR", 0.15))
    abstain_gap_floor: float = float(os.getenv("UCMD_ABS_GAP_FLOOR", 0.20))
    abstain_conflict_ceiling: float = float(os.getenv("UCMD_ABS_CONFLICT_CEILING", 0.35))
    abstain_time_evidence_relief: float = float(os.getenv("UCMD_ABS_TIME_EVIDENCE_RELIEF", 0.06))
    abstain_time_gap_relief: float = float(os.getenv("UCMD_ABS_TIME_GAP_RELIEF", 0.08))
    abstain_time_conflict_relief: float = float(os.getenv("UCMD_ABS_TIME_CONFLICT_RELIEF", 0.10))
    abstain_time_bonus_cap: float = float(os.getenv("UCMD_ABS_TIME_BONUS_CAP", 0.08))
    abstain_time_strong_evidence_gain: float = float(os.getenv("UCMD_ABS_TIME_STRONG_EVIDENCE_GAIN", 0.10))
    abstain_time_strong_gap_gain: float = float(os.getenv("UCMD_ABS_TIME_STRONG_GAP_GAIN", 0.04))
    abstain_time_strong_conflict_gain: float = float(os.getenv("UCMD_ABS_TIME_STRONG_CONFLICT_GAIN", 0.10))
    scope_attr_penalty: float = float(os.getenv("UCMD_SCOPE_ATTR_PENALTY", 2.0))
    scope_regime_penalty: float = float(os.getenv("UCMD_SCOPE_REGIME_PENALTY", 1.5))
    scope_time_penalty_near: float = float(os.getenv("UCMD_SCOPE_TIME_PENALTY_NEAR", 0.35))
    scope_time_penalty_mid: float = float(os.getenv("UCMD_SCOPE_TIME_PENALTY_MID", 0.85))
    scope_time_penalty_far: float = float(os.getenv("UCMD_SCOPE_TIME_PENALTY_FAR", 1.0))
    scope_hard_filter_attr: bool = os.getenv("UCMD_SCOPE_HARD_FILTER_ATTR", "1") == "1"
    scope_hard_filter_regime: bool = os.getenv("UCMD_SCOPE_HARD_FILTER_REGIME", "1") == "1"
    branch_tau_0: float = float(os.getenv("UCMD_BRANCH_TAU_0", 0.55))
    branch_target_rate: float = float(os.getenv("UCMD_BRANCH_TARGET_RATE", 0.08))
    branch_tau_gain: float = float(os.getenv("UCMD_BRANCH_TAU_GAIN", 0.50))
    value_mismatch_branch_boost: float = float(os.getenv("UCMD_VALUE_BRANCH_BOOST", 1.5))
    value_mismatch_update_scale: float = float(os.getenv("UCMD_VALUE_UPDATE_SCALE", 0.25))
    debug_failure_examples: int = int(os.getenv("UCMD_DEBUG_FAILURES", 20))
    sleep_every_n_steps: int = int(os.getenv("UCMD_SLEEP_EVERY", 50))


cfg = Config()


ENTITIES = ["anna", "bob", "cara", "dave", "emma", "finn", "gina", "hugo"]
ATTRS = {
    "city": ["copenhagen", "aarhus", "odense", "aalborg"],
    "role": ["engineer", "designer", "manager", "researcher"],
    "status": ["active", "paused", "busy", "offline"],
}
REGIMES = ["work", "home", "travel", "weekend"]
SOURCES = ["chat", "email", "crm", "calendar"]
TIME_BUCKETS = ["current", "same_day", "same_week", "historical"]

RESULTS_HEADER = [
    "run_id",
    "git_commit",
    "git_parent_commit",
    "description",
    "status",
    "decision_reason",
    "samples",
    "epochs",
    "joint_answer_or_abstain_acc",
    "abstain_accuracy",
    "abstain_precision",
    "abstain_recall",
    "raw_recall_at_k",
    "latest_slice_recall",
    "latest_slice_conflict_accuracy",
    "regime_mismatch",
    "time_bucket_mismatch",
    "attribute_mismatch",
    "wrong_answer_should_abstain",
    "wrong_answer_should_answer",
    "false_abstain",
    "false_merge_rate",
    "unnecessary_branch_rate",
    "header_fake_fact_rate",
    "avg_header_value_purity",
    "avg_headers_used",
    "avg_retrieval_latency_ms",
    "hard_joint_answer_or_abstain_acc",
    "hard_abstain_accuracy",
    "hard_abstain_precision",
    "hard_abstain_recall",
    "hard_raw_recall_at_k",
    "hard_latest_slice_recall",
    "hard_latest_slice_conflict_accuracy",
    "hard_regime_mismatch",
    "hard_time_bucket_mismatch",
    "hard_attribute_mismatch",
    "hard_wrong_answer_should_abstain",
    "hard_wrong_answer_should_answer",
    "hard_false_abstain",
    "hard_false_merge_rate",
    "hard_unnecessary_branch_rate",
    "hard_header_fake_fact_rate",
    "hard_avg_header_value_purity",
    "hard_avg_headers_used",
    "hard_avg_retrieval_latency_ms",
]


def scalar(x) -> float:
    mx.eval(x)
    return float(x.item())


def tsv_safe(text: str) -> str:
    return " ".join(text.replace("\t", " ").splitlines()).strip()


def zeros(d: int):
    return mx.zeros((d,), dtype=mx.float32)


def sigmoid(x):
    return 1.0 / (1.0 + mx.exp(-x))


def l2norm(x, eps: float = 1e-6):
    return x / (mx.sqrt(mx.sum(x * x)) + eps)


def softmax1d(x):
    x = x - mx.max(x)
    ex = mx.exp(x)
    return ex / mx.sum(ex)


def bce_with_logits(logit, target: float):
    t = mx.array(target, dtype=mx.float32)
    return mx.maximum(logit, 0.0) - logit * t + mx.log1p(mx.exp(-mx.abs(logit)))


def cross_entropy(logits, target: int):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -log_probs[target]


def topk_by_score(items: List, score_fn, k: int):
    scored = [(score_fn(x), i, x) for i, x in enumerate(items)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [x for _, _, x in scored[:k]]


def grad_global_norm(grads):
    total = mx.array(0.0, dtype=mx.float32)
    for _, leaf in tree_flatten(grads):
        if leaf is None:
            continue
        total = total + mx.sum(leaf * leaf)
    return mx.sqrt(total)


def clip_grad_norm(grads, max_norm: float, eps: float = 1e-6):
    grad_norm = grad_global_norm(grads)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(max_norm, dtype=mx.float32) / (grad_norm + eps),
    )
    clipped = tree_map(lambda leaf: leaf if leaf is None else leaf * scale, grads)
    return clipped, grad_norm


def score_to_mass_scalar(score, limit: float = 20.0):
    return math.exp(max(min(scalar(score), limit), -limit))


def value_vote_logits(ds: "SynthDataset", raw_items: List[dict], raw_scores):
    floor = mx.array(-8.0, dtype=mx.float32)
    if not raw_items:
        return mx.full((len(ds.values),), -8.0, dtype=mx.float32)

    logits = []
    for value_id in range(len(ds.values)):
        mass = mx.array(0.0, dtype=mx.float32)
        for raw, score in zip(raw_items, raw_scores):
            if raw["event"]["value"] != value_id:
                continue
            clipped = mx.minimum(mx.maximum(score, -20.0), 20.0)
            mass = mass + mx.exp(clipped)
        logits.append(mx.maximum(mx.log(mass + 1e-8), floor))
    return mx.stack(logits)


def value_mass_stats(ds: "SynthDataset", raw_items: List[dict], raw_scores):
    masses = [0.0] * len(ds.values)
    for raw, score in zip(raw_items, raw_scores):
        value_id = raw["event"]["value"]
        masses[value_id] += score_to_mass_scalar(score)

    ordered = sorted(masses, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    gap = top1 - top2
    total = sum(masses) + 1e-8
    conflict_mass = top2 / total
    return top1, gap, conflict_mass, masses


def scope_filtered_raw_candidates(raw_candidates: List[dict], query: dict):
    filtered = list(raw_candidates)
    if cfg.scope_hard_filter_attr:
        attr_filtered = [raw for raw in filtered if raw["event"]["attribute"] == query["attribute"]]
        if attr_filtered:
            filtered = attr_filtered
    if cfg.scope_hard_filter_regime:
        regime_filtered = [raw for raw in filtered if raw["event"]["regime"] == query["regime"]]
        if regime_filtered:
            filtered = regime_filtered
    return filtered


def time_bucket_name(thing: dict) -> Optional[str]:
    raw = thing.get("raw")
    if isinstance(raw, dict) and "time_bucket" in raw:
        return raw["time_bucket"]
    return None


def time_bucket_penalty(query_thing: dict, raw_thing: dict) -> float:
    rank = {
        "historical": 0,
        "same_week": 1,
        "same_day": 2,
        "current": 3,
    }
    query_bucket = time_bucket_name(query_thing)
    raw_bucket = time_bucket_name(raw_thing)
    if query_bucket not in rank or raw_bucket not in rank:
        return 0.0
    distance = abs(rank[query_bucket] - rank[raw_bucket])
    if distance <= 0:
        return 0.0
    if distance == 1:
        return cfg.scope_time_penalty_near
    if distance == 2:
        return cfg.scope_time_penalty_mid
    return cfg.scope_time_penalty_far


def scope_penalty(query: dict, raw_event: dict) -> float:
    penalty = 0.0
    if raw_event["attribute"] != query["attribute"]:
        penalty += cfg.scope_attr_penalty
    if raw_event["regime"] != query["regime"]:
        penalty += cfg.scope_regime_penalty
    penalty += time_bucket_penalty(query, raw_event)
    return penalty


def scope_penalized_score(score, query: dict, raw_event: dict):
    penalty = scope_penalty(query, raw_event)
    if penalty == 0.0:
        return score
    return score - mx.array(penalty, dtype=mx.float32)


def calibrated_abstain_thresholds(evidence_strength: float, gap: float, conflict_mass: float, top_time_penalty: float):
    evidence_floor = max(0.0, cfg.abstain_evidence_floor - cfg.abstain_time_evidence_relief * top_time_penalty)
    gap_floor = max(0.0, cfg.abstain_gap_floor - cfg.abstain_time_gap_relief * top_time_penalty)
    conflict_ceiling = min(
        0.99,
        cfg.abstain_conflict_ceiling + cfg.abstain_time_conflict_relief * top_time_penalty,
    )

    time_scale = min(1.0, top_time_penalty / max(cfg.scope_time_penalty_far, 1e-8)) if top_time_penalty > 0.0 else 0.0
    if time_scale > 0.0:
        strong_evidence = max(0.0, evidence_strength - 0.45)
        strong_gap = max(0.0, gap - 0.30)
        low_conflict = max(0.0, 0.30 - conflict_mass)
        extra_relief = min(
            cfg.abstain_time_bonus_cap,
            time_scale
            * (
                cfg.abstain_time_strong_evidence_gain * strong_evidence
                + cfg.abstain_time_strong_gap_gain * strong_gap
                + cfg.abstain_time_strong_conflict_gain * low_conflict
            ),
        )
        evidence_floor = max(0.0, evidence_floor - extra_relief)
        gap_floor = max(0.0, gap_floor - 0.5 * extra_relief)
        conflict_ceiling = min(0.99, conflict_ceiling + 0.5 * extra_relief)

    return evidence_floor, gap_floor, conflict_ceiling


def same_answer_scope(raw_event: dict, query: dict) -> bool:
    return (
        raw_event["entity"] == query["entity"]
        and raw_event["attribute"] == query["attribute"]
        and raw_event["regime"] == query["regime"]
    )


def select_answer_evidence(query: dict, raw_items: List[dict], raw_scores):
    scoped = [
        (raw, score)
        for raw, score in zip(raw_items, raw_scores)
        if same_answer_scope(raw["event"], query)
    ]
    if not scoped:
        return [], []

    query_bucket = time_bucket_name(query)
    if query_bucket == "current":
        latest_t = max(raw["event"]["timestamp"] for raw, _ in scoped)
        selected = [(raw, score) for raw, score in scoped if raw["event"]["timestamp"] == latest_t]
    elif query_bucket == "same_day":
        selected = [
            (raw, score)
            for raw, score in scoped
            if time_bucket_name(raw["event"]) in {"current", "same_day"}
        ]
    elif query_bucket == "same_week":
        selected = [
            (raw, score)
            for raw, score in scoped
            if time_bucket_name(raw["event"]) in {"current", "same_day", "same_week"}
        ]
    else:
        selected = scoped

    if not selected:
        selected = scoped

    selected.sort(key=lambda item: scalar(item[1]), reverse=True)
    return [raw for raw, _ in selected], [score for _, score in selected]


def time_bucket_from_step(step: int, total: int) -> str:
    if step >= total - 2:
        return "current"
    if step >= total - 4:
        return "same_day"
    if step >= total - 7:
        return "same_week"
    return "historical"


def latest_resolution(events: List[dict], entity: str, attr: str, regime: str):
    scoped = [
        event
        for event in events
        if event["entity"] == entity
        and event["attribute"] == attr
        and event["regime"] == regime
    ]
    if not scoped:
        return None, 1, []
    latest_t = max(event["timestamp"] for event in scoped)
    latest = [event for event in scoped if event["timestamp"] == latest_t]
    values = sorted({event["value"] for event in latest})
    if len(values) == 1:
        positive_ids = [event["id"] for event in latest if event["value"] == values[0]]
        return values[0], 0, positive_ids
    return None, 1, [event["id"] for event in latest]


def random_other(items: List[str], avoid: str) -> str:
    return random.choice([item for item in items if item != avoid])


def make_synth_event(
    sample_idx: int,
    event_idx: int,
    entity: str,
    attr: str,
    value: str,
    regime: str,
    timestamp: int,
    branch_target: int = 0,
    note: str = "main",
):
    text = f"{entity} {attr} is {value} under {regime}"
    if note == "conflict":
        text = f"Conflict: {entity} {attr} may be {value} under {regime}"
    return {
        "id": f"s{sample_idx}_e{event_idx}",
        "entity": entity,
        "attribute": attr,
        "value": value,
        "regime": regime,
        "timestamp": timestamp,
        "time_bucket": None,
        "source": random.choice(SOURCES),
        "text": text,
        "provenance": {"kind": "observation", "tool": None, "actor": "synthetic"},
        "branch_target": int(branch_target),
        "note": note,
    }


def finalize_sample(events: List[dict], entity: str, attr: str, query_regime: str, scenario: str):
    events.sort(key=lambda item: (item["timestamp"], item["id"]))
    total_steps = len(events)
    for i, event in enumerate(events):
        event["time_bucket"] = time_bucket_from_step(i, total_steps)

    answer, abstain, positive_ids = latest_resolution(events, entity, attr, query_regime)
    return {
        "events": events,
        "query": {
            "entity": entity,
            "attribute": attr,
            "regime": query_regime,
            "time_bucket": "current",
            "text": f"What is {entity}'s {attr} in regime {query_regime}?",
        },
        "answer": answer,
        "abstain": abstain,
        "positive_raw_ids": positive_ids,
        "scenario": scenario,
    }


def make_hard_temporal_override_sample(idx: int):
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    query_regime = random.choice(REGIMES)
    stale_value, fresh_value = random.sample(ATTRS[attr], 2)
    other_regime = random_other(REGIMES, query_regime)
    events = []
    event_idx = 0

    for timestamp in range(1, 6):
        events.append(make_synth_event(idx, event_idx, entity, attr, stale_value, query_regime, timestamp, 0))
        event_idx += 1
    for timestamp in range(6, 9):
        noise_entity = random_other(ENTITIES, entity)
        noise_attr = random.choice(sorted(ATTRS.keys()))
        noise_value = random.choice(ATTRS[noise_attr])
        noise_regime = random.choice(REGIMES)
        events.append(make_synth_event(idx, event_idx, noise_entity, noise_attr, noise_value, noise_regime, timestamp, 0, "distractor"))
        event_idx += 1
    events.append(make_synth_event(idx, event_idx, entity, attr, stale_value, other_regime, 9, 1, "cross_regime"))
    event_idx += 1
    events.append(make_synth_event(idx, event_idx, entity, attr, fresh_value, query_regime, 10, 1))
    return finalize_sample(events, entity, attr, query_regime, "temporal_override")


def make_hard_latest_conflict_sample(idx: int):
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    query_regime = random.choice(REGIMES)
    base_value, alt_value = random.sample(ATTRS[attr], 2)
    events = []
    event_idx = 0

    for timestamp in range(1, 6):
        events.append(make_synth_event(idx, event_idx, entity, attr, base_value, query_regime, timestamp, 0))
        event_idx += 1
    events.append(make_synth_event(idx, event_idx, entity, attr, base_value, query_regime, 10, 1, "conflict"))
    event_idx += 1
    events.append(make_synth_event(idx, event_idx, entity, attr, alt_value, query_regime, 10, 1, "conflict"))
    return finalize_sample(events, entity, attr, query_regime, "latest_conflict")


def make_hard_missing_recent_scope_sample(idx: int):
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    query_regime = random.choice(REGIMES)
    target_value = random.choice(ATTRS[attr])
    other_regime = random_other(REGIMES, query_regime)
    events = []
    event_idx = 0

    for timestamp in range(1, 4):
        events.append(make_synth_event(idx, event_idx, entity, attr, target_value, query_regime, timestamp, 0))
        event_idx += 1
    for timestamp in range(4, 11):
        if timestamp % 2 == 0:
            noise_value = random_other(ATTRS[attr], target_value)
            events.append(make_synth_event(idx, event_idx, entity, attr, noise_value, other_regime, timestamp, 1, "cross_regime"))
        else:
            noise_entity = random_other(ENTITIES, entity)
            noise_attr = random.choice(sorted(ATTRS.keys()))
            noise_value = random.choice(ATTRS[noise_attr])
            noise_regime = random.choice(REGIMES)
            events.append(make_synth_event(idx, event_idx, noise_entity, noise_attr, noise_value, noise_regime, timestamp, 0, "distractor"))
        event_idx += 1
    return finalize_sample(events, entity, attr, query_regime, "missing_recent_scope")


def make_hard_near_regime_sample(idx: int):
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    query_regime = random.choice(REGIMES)
    alt_regime = random_other(REGIMES, query_regime)
    target_value, alt_value = random.sample(ATTRS[attr], 2)
    events = []
    event_idx = 0

    for timestamp in range(1, 6):
        events.append(make_synth_event(idx, event_idx, entity, attr, target_value, query_regime, timestamp, 0))
        event_idx += 1
    for timestamp in range(6, 11):
        branch_target = 1 if timestamp == 6 else 0
        events.append(make_synth_event(idx, event_idx, entity, attr, alt_value, alt_regime, timestamp, branch_target, "cross_regime"))
        event_idx += 1
    return finalize_sample(events, entity, attr, query_regime, "near_regime")


def make_hard_no_scope_evidence_sample(idx: int):
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    query_regime = random.choice(REGIMES)
    alt_regime = random_other(REGIMES, query_regime)
    events = []
    event_idx = 0

    for timestamp in range(1, 6):
        value = random.choice(ATTRS[attr])
        events.append(make_synth_event(idx, event_idx, entity, attr, value, alt_regime, timestamp, 1, "cross_regime"))
        event_idx += 1
    for timestamp in range(6, 11):
        noise_attr = random_other(sorted(ATTRS.keys()), attr)
        noise_value = random.choice(ATTRS[noise_attr])
        events.append(make_synth_event(idx, event_idx, entity, noise_attr, noise_value, query_regime, timestamp, 0, "near_miss"))
        event_idx += 1
    return finalize_sample(events, entity, attr, query_regime, "no_scope_evidence")


def make_hard_thin_latest_slice_sample(idx: int):
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    query_regime = random.choice(REGIMES)
    stale_value, latest_value = random.sample(ATTRS[attr], 2)
    events = []
    event_idx = 0

    for timestamp in range(1, 8):
        events.append(make_synth_event(idx, event_idx, entity, attr, stale_value, query_regime, timestamp, 0))
        event_idx += 1
    events.append(make_synth_event(idx, event_idx, entity, attr, latest_value, query_regime, 9, 1))
    event_idx += 1
    for timestamp in range(10, 13):
        noise_entity = random_other(ENTITIES, entity)
        noise_attr = random.choice(sorted(ATTRS.keys()))
        noise_value = random.choice(ATTRS[noise_attr])
        noise_regime = random.choice(REGIMES)
        events.append(make_synth_event(idx, event_idx, noise_entity, noise_attr, noise_value, noise_regime, timestamp, 0, "distractor"))
        event_idx += 1
    return finalize_sample(events, entity, attr, query_regime, "thin_latest_slice")


HARD_SCENARIO_BUILDERS = [
    make_hard_temporal_override_sample,
    make_hard_latest_conflict_sample,
    make_hard_missing_recent_scope_sample,
    make_hard_near_regime_sample,
    make_hard_no_scope_evidence_sample,
    make_hard_thin_latest_slice_sample,
]


def make_sample(idx: int, min_events: int, max_events: int) -> dict:
    entity = random.choice(ENTITIES)
    attr = random.choice(sorted(ATTRS.keys()))
    values = ATTRS[attr]
    n_events = random.randint(min_events, max_events)
    n_shifts = random.choice([3, 4])

    current_regime = random.choice(REGIMES)
    current_value = random.choice(values)
    shift_positions = sorted(random.sample(range(1, n_events - 1), n_shifts))
    conflict_count = random.choice([1, 2])
    conflict_positions = set(random.sample(range(2, n_events - 1), conflict_count))

    events = []
    next_id = 0
    timestamp = 0

    for step in range(n_events):
        timestamp += random.choice([1, 1, 2])
        branch_target = 0

        if step in shift_positions:
            if random.random() < 0.5:
                current_regime = random.choice([item for item in REGIMES if item != current_regime])
                current_value = random.choice(values)
            else:
                current_value = random.choice([item for item in values if item != current_value])
            branch_target = 1
        elif random.random() >= 0.75:
            current_value = random.choice([item for item in values if item != current_value])
            branch_target = 1

        event = {
            "id": f"s{idx}_e{next_id}",
            "entity": entity,
            "attribute": attr,
            "value": current_value,
            "regime": current_regime,
            "timestamp": timestamp,
            "time_bucket": None,
            "source": random.choice(SOURCES),
            "text": f"{entity} {attr} is {current_value} under {current_regime}",
            "provenance": {"kind": "observation", "tool": None, "actor": "synthetic"},
            "branch_target": branch_target,
            "note": "main",
        }
        next_id += 1
        events.append(event)

        if random.random() < 0.35:
            d_entity = random.choice([item for item in ENTITIES if item != entity])
            d_attr = random.choice(sorted(ATTRS.keys()))
            d_value = random.choice(ATTRS[d_attr])
            distractor = {
                "id": f"s{idx}_e{next_id}",
                "entity": d_entity,
                "attribute": d_attr,
                "value": d_value,
                "regime": random.choice(REGIMES),
                "timestamp": timestamp,
                "time_bucket": None,
                "source": random.choice(SOURCES),
                "text": f"{d_entity} {d_attr} is {d_value}",
                "provenance": {"kind": "observation", "tool": None, "actor": "synthetic"},
                "branch_target": 0,
                "note": "distractor",
            }
            next_id += 1
            events.append(distractor)

        if step in conflict_positions:
            alt_value = random.choice([item for item in values if item != current_value])
            conflict = {
                "id": f"s{idx}_e{next_id}",
                "entity": entity,
                "attribute": attr,
                "value": alt_value,
                "regime": current_regime,
                "timestamp": timestamp,
                "time_bucket": None,
                "source": random.choice(SOURCES),
                "text": f"Conflict: {entity} {attr} may be {alt_value} under {current_regime}",
                "provenance": {"kind": "observation", "tool": None, "actor": "synthetic"},
                "branch_target": 1,
                "note": "conflict",
            }
            next_id += 1
            events.append(conflict)

    events.sort(key=lambda item: (item["timestamp"], item["id"]))
    total_steps = len(events)
    for i, event in enumerate(events):
        event["time_bucket"] = time_bucket_from_step(i, total_steps)

    query_regime = random.choice(
        sorted(
            {
                event["regime"]
                for event in events
                if event["entity"] == entity and event["attribute"] == attr
            }
        )
    )
    answer, abstain, positive_ids = latest_resolution(events, entity, attr, query_regime)
    return {
        "events": events,
        "query": {
            "entity": entity,
            "attribute": attr,
            "regime": query_regime,
            "time_bucket": "current",
            "text": f"What is {entity}'s {attr} in regime {query_regime}?",
        },
        "answer": answer,
        "abstain": abstain,
        "positive_raw_ids": positive_ids,
    }


def load_samples_from_path(path: str) -> List[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_samples_to_path(path: str, samples: List[dict]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def build_hard_eval_samples(config: Config) -> List[dict]:
    if config.hard_eval_samples <= 0:
        return []

    samples = []
    for idx in range(config.hard_eval_samples):
        builder = HARD_SCENARIO_BUILDERS[idx % len(HARD_SCENARIO_BUILDERS)]
        samples.append(builder(10_000 + idx))
    return samples


def build_samples(config: Config) -> List[dict]:
    if config.dataset_path and os.path.exists(config.dataset_path):
        return load_samples_from_path(config.dataset_path)

    samples = [make_sample(i, config.min_events, config.max_events) for i in range(config.n_samples)]
    if config.dataset_path:
        dump_samples_to_path(config.dataset_path, samples)
    return samples


class SynthDataset:
    def __init__(self, samples: List[dict]):
        self.samples = list(samples)
        random.shuffle(self.samples)

        self.entities = sorted(ENTITIES)
        self.attrs = sorted(ATTRS.keys())
        self.values = sorted({value for items in ATTRS.values() for value in items})
        self.regimes = sorted(REGIMES)
        self.sources = sorted(SOURCES)
        self.time_buckets = sorted(TIME_BUCKETS)

        self.entity2id = {item: i for i, item in enumerate(self.entities)}
        self.attr2id = {item: i for i, item in enumerate(self.attrs)}
        self.value2id = {item: i for i, item in enumerate(self.values)}
        self.regime2id = {item: i for i, item in enumerate(self.regimes)}
        self.source2id = {item: i for i, item in enumerate(self.sources)}
        self.time_bucket2id = {item: i for i, item in enumerate(self.time_buckets)}

    def vocab_payload(self) -> dict:
        return {
            "entities": self.entities,
            "attributes": self.attrs,
            "values": self.values,
            "regimes": self.regimes,
            "sources": self.sources,
            "time_buckets": self.time_buckets,
        }

    def encode_event(self, event: dict) -> dict:
        return {
            "id": event["id"],
            "entity": self.entity2id[event["entity"]],
            "attribute": self.attr2id[event["attribute"]],
            "value": self.value2id[event["value"]],
            "regime": self.regime2id[event["regime"]],
            "source": self.source2id[event["source"]],
            "time_bucket": self.time_bucket2id[event["time_bucket"]],
            "timestamp": event["timestamp"],
            "branch_target": int(event.get("branch_target", 0)),
            "raw": event,
        }

    def encode_query(self, query: dict) -> dict:
        return {
            "entity": self.entity2id[query["entity"]],
            "attribute": self.attr2id[query["attribute"]],
            "regime": self.regime2id[query["regime"]],
            "time_bucket": self.time_bucket2id[query["time_bucket"]],
            "raw": query,
        }

    def encode_sample(self, sample: dict) -> dict:
        answer = -1 if sample["answer"] is None else self.value2id[sample["answer"]]
        return {
            "events": [self.encode_event(event) for event in sample["events"]],
            "query": self.encode_query(sample["query"]),
            "answer": answer,
            "abstain": sample["abstain"],
            "positive_raw_ids": sample["positive_raw_ids"],
            "raw": sample,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.encode_sample(self.samples[idx])


def batch_iter(indices: List[int], batch_size: int, shuffle: bool = True):
    order = list(indices)
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield order[i:i + batch_size]


class FixedFeatureCodec:
    def __init__(self, ds: SynthDataset, d_field: int):
        self.ds = ds
        self.d_field = d_field
        self.tables = {
            "entity": self._make_table(len(ds.entities), d_field, 11),
            "attribute": self._make_table(len(ds.attrs), d_field, 13),
            "value": self._make_table(len(ds.values), d_field, 17),
            "regime": self._make_table(len(ds.regimes), d_field, 19),
            "source": self._make_table(len(ds.sources), d_field, 23),
            "time_bucket": self._make_table(len(ds.time_buckets), d_field, 29),
        }
        self.event_dim = 6 * d_field + 2
        self.scope_dim = 4 * d_field

    def _make_table(self, n: int, d: int, seed: int):
        rng = random.Random(SEED + seed)
        rows = []
        for _ in range(n):
            rows.append([rng.gauss(0.0, 1.0 / math.sqrt(d)) for _ in range(d)])
        return mx.array(rows, dtype=mx.float32)

    def _time_features(self, timestamp: int):
        x = float(timestamp)
        return mx.array([math.sin(x / 7.0), math.cos(x / 7.0)], dtype=mx.float32)

    def event_features(self, event: dict):
        parts = [
            self.tables["entity"][event["entity"]],
            self.tables["attribute"][event["attribute"]],
            self.tables["value"][event["value"]],
            self.tables["regime"][event["regime"]],
            self.tables["source"][event["source"]],
            self.tables["time_bucket"][event["time_bucket"]],
            self._time_features(event["timestamp"]),
        ]
        return mx.concatenate(parts, axis=0)

    def scope_features(self, thing: dict):
        parts = [
            self.tables["entity"][thing["entity"]],
            self.tables["attribute"][thing["attribute"]],
            self.tables["regime"][thing["regime"]],
            self.tables["time_bucket"][thing["time_bucket"]],
        ]
        return mx.concatenate(parts, axis=0)


class TwoLayerMLP(nn.Module):
    def __init__(self, din: int, dh: int, dout: int):
        super().__init__()
        self.l1 = nn.Linear(din, dh)
        self.l2 = nn.Linear(dh, dout)

    def __call__(self, x):
        return self.l2(nn.gelu(self.l1(x)))


class UCMDv31(nn.Module):
    def __init__(self, codec: FixedFeatureCodec, ds: SynthDataset):
        super().__init__()
        D = cfg.d_model
        H = cfg.d_hidden
        self.codec = codec
        self.ds = ds
        self.d_model = D

        self.event_mlp = TwoLayerMLP(codec.event_dim, H, D)
        self.scope_mlp = TwoLayerMLP(codec.scope_dim, H, D)
        self.query_mlp = TwoLayerMLP(codec.scope_dim, H, D)

        pair_dim = 6 * D
        self.conflict_mlp = TwoLayerMLP(pair_dim, H, 1)
        self.update_mlp = TwoLayerMLP(pair_dim, H, 1)
        self.branch_mlp = TwoLayerMLP(5, H, 1)
        self.rerank_mlp = TwoLayerMLP(pair_dim, H, 1)
        self.fuse_mlp = TwoLayerMLP(4 * D + 1, H, D)
        self.answer_head = nn.Linear(D, len(ds.values))
        self.abstain_mlp = TwoLayerMLP(D + 1, H, 1)

        self.scope_entity = nn.Linear(D, len(ds.entities))
        self.scope_attr = nn.Linear(D, len(ds.attrs))
        self.scope_regime = nn.Linear(D, len(ds.regimes))
        self.scope_time = nn.Linear(D, len(ds.time_buckets))

    def encode_event(self, event: dict):
        return l2norm(self.event_mlp(self.codec.event_features(event)))

    def encode_scope(self, thing: dict):
        return l2norm(self.scope_mlp(self.codec.scope_features(thing)))

    def encode_query(self, query: dict):
        return l2norm(self.query_mlp(self.codec.scope_features(query)))

    def pair_features(self, a, b, ua, ub):
        return mx.concatenate([a, b, mx.abs(a - b), ua, ub, mx.abs(ua - ub)], axis=0)

    def conflict_score(self, e, h, u, hu):
        return sigmoid(self.conflict_mlp(self.pair_features(e, h, u, hu))[0])

    def update_gate(self, e, h, u, hu):
        return sigmoid(self.update_mlp(self.pair_features(e, h, u, hu))[0])

    def branch_logits(self, conflict_score, match_score, budget_pressure, support_norm, quality):
        fixed = mx.array(
            [match_score, budget_pressure, support_norm, quality],
            dtype=mx.float32,
        )
        x = mx.concatenate([mx.reshape(conflict_score, (1,)), fixed], axis=0)
        return self.branch_mlp(x)[0]

    def branch_score(self, conflict_score, match_score, budget_pressure, support_norm, quality):
        return sigmoid(
            self.branch_logits(conflict_score, match_score, budget_pressure, support_norm, quality)
        )

    def rerank_score(self, q, e, uq, u):
        return self.rerank_mlp(self.pair_features(q, e, uq, u))[0]

    def fuse(self, q, r_raw, r_hdr, r_sem, evidence_strength):
        x = mx.concatenate([q, r_raw, r_hdr, r_sem, mx.array([evidence_strength], dtype=mx.float32)], axis=0)
        return self.fuse_mlp(x)

    def abstain_logit(self, z, evidence_strength):
        x = mx.concatenate([z, mx.array([evidence_strength], dtype=mx.float32)], axis=0)
        return self.abstain_mlp(x)[0]

    def abstain_prob(self, z, evidence_strength):
        return sigmoid(self.abstain_logit(z, evidence_strength))

    def scope_loss(self, u_vec, thing: dict):
        return (
            cross_entropy(self.scope_entity(u_vec), thing["entity"])
            + cross_entropy(self.scope_attr(u_vec), thing["attribute"])
            + cross_entropy(self.scope_regime(u_vec), thing["regime"])
            + cross_entropy(self.scope_time(u_vec), thing["time_bucket"])
        ) / 4.0


def scope_key(thing: dict) -> Tuple[int, int, int, int]:
    return (thing["entity"], thing["attribute"], thing["regime"], thing["time_bucket"])


def relaxed_scope_keys(thing: dict):
    entity = thing["entity"]
    attribute = thing["attribute"]
    regime = thing["regime"]
    time_bucket = thing["time_bucket"]
    return [
        (entity, attribute, regime, time_bucket),
        (entity, attribute, regime, -1),
        (entity, attribute, -1, -1),
    ]


def new_state(d_model: int):
    return {
        "raw_hot": [],
        "raw_cold": [],
        "raw_lookup": {},
        "cold_scope_index": {},
        "headers": [],
        "semantic_state": zeros(d_model),
        "branch_rate_ema": 0.0,
        "step": 0,
    }


def header_quality(header: dict):
    support = math.log(1.0 + header["support_count"])
    conflict = header["conflict_count"]
    variance = header["variance"]
    x = 1.2 * support - 1.5 * conflict - 1.0 * variance
    return 1.0 / (1.0 + math.exp(-x))


def update_header_stats(header: dict, e, conflict_prob: float):
    old_h = header["h"]
    header["variance"] = 0.95 * header["variance"] + 0.05 * abs(scalar(mx.mean(mx.abs(e - old_h))))
    header["conflict_count"] += float(conflict_prob > 0.5)
    header["quality"] = header_quality(header)


def semantic_from_headers(state: dict):
    mature = [header for header in state["headers"] if header["quality"] >= cfg.quality_tau]
    if not mature:
        return state["semantic_state"]

    total = sum(header["quality"] for header in mature) + 1e-8
    pooled = zeros(state["semantic_state"].shape[0])
    for header in mature:
        pooled = pooled + (header["quality"] / total) * header["h"]
    return (1.0 - cfg.semantic_beta) * state["semantic_state"] + cfg.semantic_beta * pooled


def adaptive_branch_tau(state: dict):
    return cfg.branch_tau_0 + cfg.branch_tau_gain * (
        state["branch_rate_ema"] - cfg.branch_target_rate
    )


def register_cold_item(state: dict, raw_item: dict):
    for key in relaxed_scope_keys(raw_item["event"]):
        bucket = state["cold_scope_index"].setdefault(key, [])
        bucket.append(raw_item["raw_id"])


def find_best_header(headers: List[dict], e, u):
    if not headers:
        return None, 0.0, None

    best = None
    best_score = -1e9
    best_idx = None
    for i, header in enumerate(headers):
        score = scalar(mx.sum(e * header["h"]) + 0.5 * mx.sum(u * header["u"]))
        if not math.isfinite(score):
            score = -1e9
        if best is None or score > best_score:
            best = header
            best_score = score
            best_idx = i
    scaled_score = best_score
    scaled_score = max(min(scaled_score, 30.0), -30.0)
    match_score = 1.0 / (1.0 + math.exp(-scaled_score))
    return best_idx, match_score, best


def make_header(header_id: str, e, u, raw_id: str, conflict_prob: float, value_id: int):
    value_id = int(value_id)
    header = {
        "header_id": header_id,
        "h": e,
        "u": u,
        "support_count": 1,
        "conflict_count": float(conflict_prob > 0.5),
        "variance": 0.0,
        "quality": 0.55,
        "backlinks": [raw_id],
        "value_hist": {value_id: 1},
        "dominant_value": value_id,
    }
    header["quality"] = header_quality(header)
    return header


def add_observation_to_state(model: UCMDv31, state: dict, event: dict, teacher_force: bool = False):
    e_live = model.encode_event(event)
    u_live = model.encode_scope(event)
    scope_loss = model.scope_loss(u_live, event)
    e = mx.stop_gradient(e_live)
    u = mx.stop_gradient(u_live)

    raw_item = {
        "raw_id": event["id"],
        "event": event,
        "e": e,
        "u": u,
    }
    state["raw_hot"].append(raw_item)
    state["raw_lookup"][raw_item["raw_id"]] = raw_item

    if len(state["raw_hot"]) > cfg.n_hot:
        moved = state["raw_hot"].pop(0)
        state["raw_cold"].append(moved)
        register_cold_item(state, moved)

    if not state["headers"]:
        header_id = "hdr_0"
        state["headers"].append(make_header(header_id, e, u, event["id"], 1.0, event["value"]))
        state["branch_rate_ema"] = 0.95 * state["branch_rate_ema"] + 0.05 * 1.0
        state["step"] += 1
        return {
            "scope_loss": scope_loss,
            "branch_loss": bce_with_logits(mx.array(6.0, dtype=mx.float32), 1.0),
            "conflict_score": mx.array(1.0, dtype=mx.float32),
            "branch_prob": mx.array(1.0, dtype=mx.float32),
            "branched": True,
            "header_id": header_id,
        }

    best_idx, match_score, best = find_best_header(state["headers"], e_live, u_live)
    conflict = model.conflict_score(e_live, best["h"], u_live, best["u"])
    budget = len(state["headers"]) / float(cfg.h_max)
    support_norm = min(1.0, best["support_count"] / 8.0)
    quality = best["quality"]
    incoming_value = int(event["value"])
    dominant_value = int(best.get("dominant_value", -1))
    value_mismatch = float(dominant_value >= 0 and incoming_value != dominant_value)
    branch_logit = model.branch_logits(conflict, match_score, budget, support_norm, quality)
    branch_logit = branch_logit + cfg.value_mismatch_branch_boost * mx.array(value_mismatch, dtype=mx.float32)
    branch_prob = sigmoid(branch_logit)
    branch_loss = bce_with_logits(branch_logit, float(event["branch_target"]))

    should_branch = bool(event["branch_target"]) if teacher_force else bool(
        scalar(branch_prob) > adaptive_branch_tau(state)
    )

    header_id = best["header_id"]
    if should_branch:
        header_id = f"hdr_{state['step']}"
        new_header = make_header(header_id, e, u, event["id"], scalar(conflict), incoming_value)
        if len(state["headers"]) < cfg.h_max:
            state["headers"].append(new_header)
        else:
            replace_idx = min(range(len(state["headers"])), key=lambda i: state["headers"][i]["quality"])
            state["headers"][replace_idx] = new_header
        state["branch_rate_ema"] = 0.95 * state["branch_rate_ema"] + 0.05 * 1.0
    else:
        alpha = model.update_gate(e_live, best["h"], u_live, best["u"])
        if value_mismatch:
            alpha = alpha * cfg.value_mismatch_update_scale
        best["h"] = mx.stop_gradient(l2norm((1.0 - alpha) * best["h"] + alpha * e))
        best["u"] = mx.stop_gradient(l2norm((1.0 - alpha) * best["u"] + alpha * u))
        best["support_count"] += 1
        best["backlinks"].append(event["id"])
        best.setdefault("value_hist", {})
        best["value_hist"][incoming_value] = best["value_hist"].get(incoming_value, 0) + 1
        best["dominant_value"] = max(best["value_hist"], key=best["value_hist"].get)
        update_header_stats(best, e, scalar(conflict))
        state["branch_rate_ema"] = 0.95 * state["branch_rate_ema"] + 0.05 * 0.0

    state["step"] += 1
    return {
        "scope_loss": scope_loss,
        "branch_loss": branch_loss,
        "conflict_score": conflict,
        "branch_prob": branch_prob,
        "branched": should_branch,
        "header_id": header_id,
    }


def pool_vectors(weights, vectors, d_model: int):
    if not vectors:
        return zeros(d_model)
    matrix = mx.stack(vectors, axis=0)
    return mx.sum(weights[:, None] * matrix, axis=0)


def retrieve_from_state(model: UCMDv31, state: dict, query: dict, positive_raw_ids: Optional[List[str]] = None):
    q = model.encode_query(query)
    uq = model.encode_scope(query)

    hot_candidates = topk_by_score(
        state["raw_hot"],
        lambda raw: scalar(mx.sum(uq * raw["u"])),
        cfg.k_scope,
    )

    header_candidates = topk_by_score(
        state["headers"],
        lambda header: scalar(mx.sum(q * header["h"]) + 0.5 * mx.sum(uq * header["u"])),
        min(4, len(state["headers"])),
    )

    cold_ids = []
    seen_ids = set()
    for key in relaxed_scope_keys(query):
        for raw_id in state["cold_scope_index"].get(key, []):
            if raw_id not in seen_ids:
                seen_ids.add(raw_id)
                cold_ids.append(raw_id)

    for header in header_candidates:
        for raw_id in header["backlinks"]:
            if raw_id in state["raw_lookup"] and raw_id not in seen_ids:
                seen_ids.add(raw_id)
                cold_ids.append(raw_id)

    cold_candidates = [state["raw_lookup"][raw_id] for raw_id in cold_ids if raw_id in state["raw_lookup"]]
    cold_candidates = topk_by_score(
        cold_candidates,
        lambda raw: scalar(mx.sum(uq * raw["u"])),
        cfg.k_scope,
    )

    raw_candidates = []
    merged_seen = set()
    for raw in hot_candidates + cold_candidates:
        if raw["raw_id"] not in merged_seen:
            merged_seen.add(raw["raw_id"])
            raw_candidates.append(raw)
    raw_candidates = scope_filtered_raw_candidates(raw_candidates, query)

    if raw_candidates:
        raw_base_scores = [model.rerank_score(q, raw["e"], uq, raw["u"]) for raw in raw_candidates]
        raw_scores = [
            scope_penalized_score(score, query, raw["event"])
            for raw, score in zip(raw_candidates, raw_base_scores)
        ]
        order = sorted(
            range(len(raw_candidates)),
            key=lambda idx: scalar(raw_scores[idx]),
            reverse=True,
        )[: cfg.k_rerank]
        raw_final = [raw_candidates[idx] for idx in order]
        final_score_list = [raw_scores[idx] for idx in order]
        final_score_values = [scalar(score) for score in final_score_list]
        final_scores = mx.stack(final_score_list)
        weights = softmax1d(final_scores)
        r_raw = pool_vectors(weights, [raw["e"] for raw in raw_final], model.d_model)
    else:
        raw_final = []
        final_score_list = []
        final_score_values = []
        final_scores = None
        r_raw = zeros(model.d_model)

    if header_candidates:
        header_scores = mx.stack(
            [mx.sum(q * header["h"]) + 0.5 * mx.sum(uq * header["u"]) for header in header_candidates]
        )
        header_weights = softmax1d(header_scores)
        r_hdr = pool_vectors(header_weights, [header["h"] for header in header_candidates], model.d_model)
    else:
        r_hdr = zeros(model.d_model)

    answer_raw, answer_score_list = select_answer_evidence(query, raw_candidates, raw_scores if raw_candidates else [])
    answer_score_values = [scalar(score) for score in answer_score_list]
    value_logits = value_vote_logits(model.ds, answer_raw, answer_score_list)
    top_mass, gap, conflict_mass, value_masses = value_mass_stats(model.ds, answer_raw, answer_score_list)
    total_mass = sum(value_masses) + 1e-8
    evidence_strength = top_mass / total_mass
    top_time_penalty = (
        time_bucket_penalty(query, answer_raw[0]["event"])
        if answer_raw
        else 0.0
    )
    evidence_floor, gap_floor, conflict_ceiling = calibrated_abstain_thresholds(
        evidence_strength,
        gap,
        conflict_mass,
        top_time_penalty,
    )
    rule_abstain = (
        evidence_strength < evidence_floor
        or gap < gap_floor
        or conflict_mass > conflict_ceiling
    )

    r_sem = semantic_from_headers(state)
    z = model.fuse(q, r_raw, r_hdr, r_sem, evidence_strength)
    latent_answer_logits = model.answer_head(z)
    abstain_logit = model.abstain_logit(z, evidence_strength)
    abstain_prob = sigmoid(abstain_logit)
    final_abstain = rule_abstain or bool(scalar(abstain_prob) > cfg.abstain_tau)

    retrieval_loss = None
    if positive_raw_ids and raw_candidates:
        pos_ids = set(positive_raw_ids)
        raw_logits = mx.stack(raw_scores)
        pos_idx = next((i for i, raw in enumerate(raw_candidates) if raw["raw_id"] in pos_ids), None)
        if pos_idx is not None:
            retrieval_loss = cross_entropy(raw_logits, pos_idx)

    return {
        "raw": raw_final,
        "raw_score_values": final_score_values,
        "answer_raw": answer_raw,
        "answer_score_values": answer_score_values,
        "headers": header_candidates,
        "semantic": r_sem,
        "answer_logits": value_logits,
        "value_logits": value_logits,
        "latent_answer_logits": latent_answer_logits,
        "abstain_logit": abstain_logit,
        "abstain_prob": abstain_prob,
        "rule_abstain": rule_abstain,
        "abstain": final_abstain,
        "evidence_strength": evidence_strength,
        "mass_gap": gap,
        "conflict_mass": conflict_mass,
        "top_time_penalty": top_time_penalty,
        "evidence_floor": evidence_floor,
        "gap_floor": gap_floor,
        "conflict_ceiling": conflict_ceiling,
        "value_masses": value_masses,
        "retrieval_loss": retrieval_loss,
    }


def sample_loss(model: UCMDv31, sample: dict, return_parts: bool = False):
    state = new_state(model.d_model)
    branch_terms = []
    scope_terms = []

    for event in sample["events"]:
        aux = add_observation_to_state(model, state, event, teacher_force=True)
        branch_terms.append(aux["branch_loss"])
        scope_terms.append(aux["scope_loss"])

    out = retrieve_from_state(model, state, sample["query"], sample["positive_raw_ids"])

    answer_loss = mx.array(0.0, dtype=mx.float32)
    if sample["abstain"] == 0 and sample["answer"] >= 0:
        answer_loss = cross_entropy(out["value_logits"], sample["answer"])

    abstain_loss = bce_with_logits(out["abstain_logit"], float(sample["abstain"]))

    branch_loss = mx.mean(mx.stack(branch_terms)) if branch_terms else mx.array(0.0, dtype=mx.float32)
    scope_loss = mx.mean(mx.stack(scope_terms)) if scope_terms else mx.array(0.0, dtype=mx.float32)
    retrieval_loss = out["retrieval_loss"] if out["retrieval_loss"] is not None else mx.array(0.0, dtype=mx.float32)

    parts = {
        "answer_loss": answer_loss,
        "abstain_loss": abstain_loss,
        "branch_loss": branch_loss,
        "scope_loss": scope_loss,
        "retrieval_loss": retrieval_loss,
    }
    parts["total_loss"] = (
        cfg.lambda_answer * answer_loss
        + cfg.lambda_abstain * abstain_loss
        + cfg.lambda_branch * branch_loss
        + cfg.lambda_scope * scope_loss
        + cfg.lambda_retrieval * retrieval_loss
    )
    if return_parts:
        return parts
    return parts["total_loss"]


def batch_loss(model: UCMDv31, batch: List[dict]):
    totals = [sample_loss(model, sample) for sample in batch]
    return mx.mean(mx.stack(totals)) if totals else mx.array(0.0, dtype=mx.float32)


def batch_loss_parts(model: UCMDv31, batch: List[dict]):
    keys = [
        "total_loss",
        "answer_loss",
        "abstain_loss",
        "branch_loss",
        "scope_loss",
        "retrieval_loss",
    ]
    buckets = {key: [] for key in keys}
    for sample in batch:
        parts = sample_loss(model, sample, return_parts=True)
        for key in keys:
            buckets[key].append(parts[key])

    return {
        key: (mx.mean(mx.stack(values)) if values else mx.array(0.0, dtype=mx.float32))
        for key, values in buckets.items()
    }


class Memory:
    def __init__(self, model: UCMDv31, ds: SynthDataset):
        self.model = model
        self.ds = ds
        self.state = new_state(model.d_model)

    def add_observation(self, event: dict, provenance: Optional[dict] = None) -> dict:
        if provenance is not None:
            event = dict(event)
            event["provenance"] = provenance
        encoded = self.ds.encode_event(event)
        aux = add_observation_to_state(self.model, self.state, encoded, teacher_force=False)
        return {
            "raw_id": encoded["id"],
            "header_id": aux["header_id"],
            "branched": aux["branched"],
            "conflict_score": scalar(aux["conflict_score"]),
            "branch_score": scalar(aux["branch_prob"]),
        }

    def retrieve(self, query: dict, k: int = 8) -> dict:
        _ = k
        encoded_query = self.ds.encode_query(query)
        out = retrieve_from_state(self.model, self.state, encoded_query, None)
        answer_id = int(mx.argmax(out["value_logits"]).item())
        return {
            "raw": [raw["event"]["raw"] for raw in out["raw"]],
            "answer_raw": [raw["event"]["raw"] for raw in out["answer_raw"]],
            "headers": [
                {
                    "header_id": header["header_id"],
                    "support_count": header["support_count"],
                    "quality": header["quality"],
                    "dominant_value": self.ds.values[header["dominant_value"]] if header.get("dominant_value", -1) >= 0 else None,
                    "backlinks": list(header["backlinks"]),
                }
                for header in out["headers"]
            ],
            "semantic": out["semantic"].tolist(),
            "answer": self.ds.values[answer_id],
            "answer_logits": out["value_logits"].tolist(),
            "latent_answer_logits": out["latent_answer_logits"].tolist(),
            "abstain_prob": scalar(out["abstain_prob"]),
            "rule_abstain": out["rule_abstain"],
            "abstain": out["abstain"],
            "evidence_strength": out["evidence_strength"],
            "mass_gap": out["mass_gap"],
            "conflict_mass": out["conflict_mass"],
            "top_time_penalty": out["top_time_penalty"],
            "evidence_floor": out["evidence_floor"],
            "gap_floor": out["gap_floor"],
            "conflict_ceiling": out["conflict_ceiling"],
            "value_masses": list(out["value_masses"]),
        }

    def decide_after_abstain(self, query: dict, retrieval_result: dict, high_risk: bool = False) -> dict:
        _ = query
        top_time_penalty = retrieval_result.get("top_time_penalty", 0.0)
        evidence_floor, gap_floor, conflict_ceiling = calibrated_abstain_thresholds(
            retrieval_result.get("evidence_strength", 0.0),
            retrieval_result.get("mass_gap", 0.0),
            retrieval_result.get("conflict_mass", 0.0),
            top_time_penalty,
        )
        if retrieval_result.get("conflict_mass", 0.0) > conflict_ceiling:
            return {"action": "ask_human" if high_risk else "reflect_and_retry"}
        if retrieval_result.get("evidence_strength", 0.0) < evidence_floor:
            return {"action": "ask_human" if high_risk else "use_tool"}
        if retrieval_result.get("mass_gap", 0.0) < gap_floor:
            return {"action": "reflect_and_retry"}
        abstain_prob = retrieval_result["abstain_prob"]
        if abstain_prob > 0.80:
            return {"action": "ask_human" if high_risk else "use_tool"}
        if abstain_prob > cfg.abstain_tau:
            return {"action": "reflect_and_retry"}
        return {"action": "defer"}

    def consolidate(self, force: bool = False) -> dict:
        _ = force
        for header in self.state["headers"]:
            header["quality"] = header_quality(header)
        self.state["semantic_state"] = semantic_from_headers(self.state)
        return {
            "num_headers": len(self.state["headers"]),
            "num_raw_hot": len(self.state["raw_hot"]),
            "num_raw_cold": len(self.state["raw_cold"]),
        }

    def save(self, path: str) -> None:
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        meta = {
            "version": "ucmd_v3_2_baseline",
            "step": self.state["step"],
            "d_model": self.model.d_model,
            "H_max": cfg.h_max,
            "N_hot": cfg.n_hot,
        }
        (base / "meta.json").write_text(json.dumps(meta, indent=2))

        with open(base / "raw_hot.jsonl", "w") as handle:
            for raw in self.state["raw_hot"]:
                handle.write(json.dumps(raw["event"]["raw"]) + "\n")

        with open(base / "raw_cold.jsonl", "w") as handle:
            for raw in self.state["raw_cold"]:
                handle.write(json.dumps(raw["event"]["raw"]) + "\n")

        header_count = len(self.state["headers"])
        if header_count:
            h_matrix = np.stack([np.array(header["h"].tolist(), dtype=np.float32) for header in self.state["headers"]])
            u_matrix = np.stack([np.array(header["u"].tolist(), dtype=np.float32) for header in self.state["headers"]])
        else:
            h_matrix = np.zeros((0, self.model.d_model), dtype=np.float32)
            u_matrix = np.zeros((0, self.model.d_model), dtype=np.float32)

        np.savez(
            base / "headers.npz",
            h=h_matrix,
            u=u_matrix,
            support_counts=np.array([header["support_count"] for header in self.state["headers"]], dtype=np.int32),
            conflict_counts=np.array([header["conflict_count"] for header in self.state["headers"]], dtype=np.float32),
            variances=np.array([header["variance"] for header in self.state["headers"]], dtype=np.float32),
            qualities=np.array([header["quality"] for header in self.state["headers"]], dtype=np.float32),
            header_ids=np.array([header["header_id"] for header in self.state["headers"]], dtype=object),
            backlinks=np.array([json.dumps(header["backlinks"]) for header in self.state["headers"]], dtype=object),
            dominant_values=np.array([header.get("dominant_value", -1) for header in self.state["headers"]], dtype=np.int32),
            value_hists=np.array(
                [json.dumps({str(k): int(v) for k, v in header.get("value_hist", {}).items()}) for header in self.state["headers"]],
                dtype=object,
            ),
        )
        np.save(base / "semantic.npy", np.array(self.state["semantic_state"].tolist(), dtype=np.float32))
        (base / "vocab.json").write_text(json.dumps(self.ds.vocab_payload(), indent=2))

    @classmethod
    def load(cls, path: str, model: UCMDv31, ds: SynthDataset):
        memory = cls(model, ds)
        base = Path(path)
        if not base.exists():
            return memory

        memory.state = new_state(model.d_model)
        meta_path = base / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            memory.state["step"] = int(meta.get("step", 0))

        for filename, target in [("raw_hot.jsonl", "raw_hot"), ("raw_cold.jsonl", "raw_cold")]:
            file_path = base / filename
            if not file_path.exists():
                continue
            with open(file_path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw_event = json.loads(line)
                    encoded = ds.encode_event(raw_event)
                    raw_item = {
                        "raw_id": encoded["id"],
                        "event": encoded,
                        "e": mx.stop_gradient(model.encode_event(encoded)),
                        "u": mx.stop_gradient(model.encode_scope(encoded)),
                    }
                    memory.state[target].append(raw_item)
                    memory.state["raw_lookup"][raw_item["raw_id"]] = raw_item
                    if target == "raw_cold":
                        register_cold_item(memory.state, raw_item)

        headers_path = base / "headers.npz"
        if headers_path.exists():
            headers_blob = np.load(headers_path, allow_pickle=True)
            has_dominant_values = "dominant_values" in headers_blob.files
            has_value_hists = "value_hists" in headers_blob.files
            for idx, header_id in enumerate(headers_blob["header_ids"].tolist()):
                value_hist = {}
                if has_value_hists:
                    raw_hist = json.loads(headers_blob["value_hists"][idx])
                    value_hist = {int(k): int(v) for k, v in raw_hist.items()}
                backlinks = json.loads(headers_blob["backlinks"][idx])
                if not value_hist:
                    for raw_id in backlinks:
                        raw_item = memory.state["raw_lookup"].get(raw_id)
                        if raw_item is None:
                            continue
                        value_id = int(raw_item["event"]["value"])
                        value_hist[value_id] = value_hist.get(value_id, 0) + 1
                dominant_value = int(headers_blob["dominant_values"][idx]) if has_dominant_values else -1
                if dominant_value < 0 and value_hist:
                    dominant_value = max(value_hist, key=value_hist.get)
                memory.state["headers"].append(
                    {
                        "header_id": header_id,
                        "h": mx.array(headers_blob["h"][idx]),
                        "u": mx.array(headers_blob["u"][idx]),
                        "support_count": int(headers_blob["support_counts"][idx]),
                        "conflict_count": float(headers_blob["conflict_counts"][idx]),
                        "variance": float(headers_blob["variances"][idx]),
                        "quality": float(headers_blob["qualities"][idx]),
                        "backlinks": backlinks,
                        "value_hist": value_hist,
                        "dominant_value": dominant_value,
                    }
                )

        semantic_path = base / "semantic.npy"
        if semantic_path.exists():
            memory.state["semantic_state"] = mx.array(np.load(semantic_path))

        return memory


def header_fake_fact_rate(state: dict):
    mature_headers = [header for header in state["headers"] if header["quality"] >= cfg.quality_tau]
    if not mature_headers:
        return 0.0

    fake_headers = 0
    for header in mature_headers:
        values = set()
        for raw_id in header["backlinks"]:
            raw = state["raw_lookup"].get(raw_id)
            if raw is None:
                continue
            values.add(raw["event"]["raw"]["value"])
        fake_headers += int(len(values) > 1)
    return fake_headers / len(mature_headers)


def header_value_purity(header: dict, state: dict):
    counts = {}
    total = 0
    for raw_id in header["backlinks"]:
        raw = state["raw_lookup"].get(raw_id)
        if raw is None:
            continue
        value = raw["event"]["raw"]["value"]
        counts[value] = counts.get(value, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def top_supporting_raw(raw_items: List[dict], raw_score_values: List[float], value_id: Optional[int] = None):
    best = None
    best_score = -1e9
    for raw, score in zip(raw_items, raw_score_values):
        if value_id is not None and raw["event"]["value"] != value_id:
            continue
        if score > best_score:
            best = raw
            best_score = score
    if best is not None:
        return best
    if raw_items:
        return raw_items[0]
    return None


def failure_record(ds: SynthDataset, sample: dict, out: dict, state: dict, pred_abs: int, pred_answer: int):
    gold_abs = int(sample["abstain"])
    gold_answer = sample["answer"]
    if gold_abs == 1 and pred_abs == 0:
        bucket = "wrong_answer_should_abstain"
    elif gold_abs == 0 and pred_abs == 1:
        bucket = "false_abstain"
    elif gold_abs == 0 and pred_abs == 0 and pred_answer != gold_answer:
        bucket = "wrong_answer_should_answer"
    else:
        return None

    support_pool = out["answer_raw"] if out["answer_raw"] else out["raw"]
    support_scores = out["answer_score_values"] if out["answer_raw"] else out["raw_score_values"]
    support = top_supporting_raw(support_pool, support_scores, None if pred_abs else pred_answer)
    tags = []
    mature_headers = [header for header in state["headers"] if header["quality"] >= cfg.quality_tau]
    contaminated = any(header_value_purity(header, state) < 0.999 for header in mature_headers)
    if contaminated:
        tags.append("header_contamination")
    if support is not None:
        support_event = support["event"]
        if support_event["regime"] != sample["query"]["regime"]:
            tags.append("regime_mismatch")
        if support_event["time_bucket"] != sample["query"]["time_bucket"]:
            tags.append("time_bucket_mismatch")
        if support_event["attribute"] != sample["query"]["attribute"]:
            tags.append("attribute_mismatch")
    return {
        "scenario": sample["raw"].get("scenario", "standard"),
        "bucket": bucket,
        "tags": tags,
        "gold": "ABSTAIN" if gold_abs else ds.values[gold_answer],
        "pred": "ABSTAIN" if pred_abs else ds.values[pred_answer],
        "query": sample["query"]["raw"],
        "support": support["event"]["raw"] if support is not None else None,
        "evidence_strength": out["evidence_strength"],
        "mass_gap": out["mass_gap"],
        "conflict_mass": out["conflict_mass"],
        "top_time_penalty": out["top_time_penalty"],
        "evidence_floor": out["evidence_floor"],
        "gap_floor": out["gap_floor"],
        "conflict_ceiling": out["conflict_ceiling"],
        "rule_abstain": out["rule_abstain"],
        "abstain_prob": scalar(out["abstain_prob"]),
    }


def print_eval_debug(metrics: dict):
    debug = metrics.get("debug")
    if not debug:
        return

    print("\n--- eval debug ---")
    print(
        "abstain confusion:"
        f" tp={debug['abstain_tp']}"
        f" fp={debug['abstain_fp']}"
        f" tn={debug['abstain_tn']}"
        f" fn={debug['abstain_fn']}"
    )
    print(
        "failure buckets:"
        f" wrong_answer_should_abstain={debug['failure_buckets']['wrong_answer_should_abstain']}"
        f" wrong_answer_should_answer={debug['failure_buckets']['wrong_answer_should_answer']}"
        f" false_abstain={debug['failure_buckets']['false_abstain']}"
    )
    print(
        "failure tags:"
        f" header_contamination={debug['failure_tags']['header_contamination']}"
        f" regime_mismatch={debug['failure_tags']['regime_mismatch']}"
        f" time_bucket_mismatch={debug['failure_tags']['time_bucket_mismatch']}"
        f" attribute_mismatch={debug['failure_tags']['attribute_mismatch']}"
    )
    print(
        "mature headers:"
        f" count={debug['mature_header_count']}"
        f" samples_with_mature_headers={debug['samples_with_mature_headers']}"
    )
    if debug["scenario_breakdown"]:
        print("scenario breakdown:")
        for scenario, stats in sorted(debug["scenario_breakdown"].items()):
            joint_acc = stats["joint_correct"] / max(1, stats["count"])
            print(
                f"  {scenario}: count={stats['count']} joint_acc={joint_acc:.3f} "
                f"abstain_gold={stats['abstain_gold']}"
            )

    if not debug["failure_examples"]:
        return

    print("\n--- failure examples ---")
    for idx, item in enumerate(debug["failure_examples"], start=1):
        tag_text = ", ".join(item["tags"]) if item["tags"] else "none"
        print(
            f"{idx:2d}. {item['scenario']} | {item['bucket']} | gold={item['gold']} | pred={item['pred']} "
            f"| evidence={item['evidence_strength']:.3f} | gap={item['mass_gap']:.3f} "
            f"| conflict={item['conflict_mass']:.3f} | time_pen={item['top_time_penalty']:.3f} | tags={tag_text}"
        )
        print(
            f"    thresholds: evidence>={item['evidence_floor']:.3f} "
            f"gap>={item['gap_floor']:.3f} conflict<={item['conflict_ceiling']:.3f}"
        )
        print(f"    query: {item['query']}")
        if item["support"] is not None:
            print(f"    support: {item['support']}")


def evaluate(model: UCMDv31, data: List[dict], collect_debug: bool = False, max_failure_examples: int = 20):
    metrics = {
        "joint_answer_or_abstain_acc": 0.0,
        "abstain_accuracy": 0.0,
        "abstain_precision": 0.0,
        "abstain_recall": 0.0,
        "raw_recall_at_k": 0.0,
        "latest_slice_recall": 0.0,
        "latest_slice_conflict_accuracy": 0.0,
        "false_merge_rate": 0.0,
        "unnecessary_branch_rate": 0.0,
        "header_fake_fact_rate": 0.0,
        "avg_header_value_purity": 0.0,
        "avg_headers_used": 0.0,
        "avg_retrieval_latency_ms": 0.0,
    }

    total = 0
    abstain_tp = 0
    abstain_pred = 0
    abstain_gold = 0
    abstain_fp = 0
    abstain_tn = 0
    abstain_fn = 0
    branch_positive = 0
    branch_negative = 0
    false_merges = 0
    unnecessary_branches = 0
    mature_header_count = 0
    mature_header_samples = 0
    purity_sum = 0.0
    fake_headers = 0
    latest_slice_total = 0
    latest_slice_hits = 0
    latest_conflict_total = 0
    latest_conflict_hits = 0
    debug = {
        "abstain_tp": 0,
        "abstain_fp": 0,
        "abstain_tn": 0,
        "abstain_fn": 0,
        "mature_header_count": 0,
        "samples_with_mature_headers": 0,
        "failure_buckets": {
            "wrong_answer_should_abstain": 0,
            "wrong_answer_should_answer": 0,
            "false_abstain": 0,
        },
        "failure_tags": {
            "header_contamination": 0,
            "regime_mismatch": 0,
            "time_bucket_mismatch": 0,
            "attribute_mismatch": 0,
        },
        "scenario_breakdown": {},
        "failure_examples": [],
    }

    for sample in data:
        state = new_state(model.d_model)
        for event in sample["events"]:
            aux = add_observation_to_state(model, state, event, teacher_force=False)
            if event["branch_target"]:
                branch_positive += 1
                false_merges += int(not aux["branched"])
            else:
                branch_negative += 1
                unnecessary_branches += int(aux["branched"])

        t0 = time.perf_counter()
        out = retrieve_from_state(model, state, sample["query"], sample["positive_raw_ids"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_abs = int(out["abstain"])
        gold_abs = int(sample["abstain"])
        abstain_tp += int(pred_abs == 1 and gold_abs == 1)
        abstain_pred += int(pred_abs == 1)
        abstain_gold += int(gold_abs == 1)
        abstain_fp += int(pred_abs == 1 and gold_abs == 0)
        abstain_tn += int(pred_abs == 0 and gold_abs == 0)
        abstain_fn += int(pred_abs == 0 and gold_abs == 1)
        metrics["abstain_accuracy"] += int(pred_abs == gold_abs)
        metrics["avg_retrieval_latency_ms"] += latency_ms
        metrics["avg_headers_used"] += len(out["headers"])
        mature_headers = [header for header in state["headers"] if header["quality"] >= cfg.quality_tau]
        if mature_headers:
            mature_header_samples += 1
            debug["samples_with_mature_headers"] += 1
        for header in mature_headers:
            purity = header_value_purity(header, state)
            purity_sum += purity
            mature_header_count += 1
            debug["mature_header_count"] += 1
            fake_headers += int(purity < 0.999)

        pred_answer = -1 if pred_abs else int(mx.argmax(out["value_logits"]).item())
        joint_correct = 0
        if gold_abs == 1:
            joint_correct = int(pred_abs == 1)
            metrics["joint_answer_or_abstain_acc"] += joint_correct
        else:
            joint_correct = int(pred_abs == 0 and pred_answer == sample["answer"])
            metrics["joint_answer_or_abstain_acc"] += joint_correct

        if sample["positive_raw_ids"]:
            positive_set = set(sample["positive_raw_ids"])
            metrics["raw_recall_at_k"] += int(any(raw["raw_id"] in positive_set for raw in out["raw"]))
            latest_slice_total += 1
            latest_slice_hits += int(any(raw["raw_id"] in positive_set for raw in out["answer_raw"]))

        if gold_abs == 1 and sample["positive_raw_ids"]:
            latest_conflict_total += 1
            latest_conflict_hits += int(pred_abs == 1)

        if collect_debug:
            scenario = sample["raw"].get("scenario", "standard")
            stats = debug["scenario_breakdown"].setdefault(
                scenario,
                {"count": 0, "joint_correct": 0, "abstain_gold": 0},
            )
            stats["count"] += 1
            stats["joint_correct"] += joint_correct
            stats["abstain_gold"] += gold_abs
            debug["abstain_tp"] = abstain_tp
            debug["abstain_fp"] = abstain_fp
            debug["abstain_tn"] = abstain_tn
            debug["abstain_fn"] = abstain_fn
            record = failure_record(model.ds, sample, out, state, pred_abs, pred_answer)
            if record is not None:
                debug["failure_buckets"][record["bucket"]] += 1
                for tag in record["tags"]:
                    debug["failure_tags"][tag] += 1
                if len(debug["failure_examples"]) < max_failure_examples:
                    debug["failure_examples"].append(record)

        total += 1

    metrics["joint_answer_or_abstain_acc"] /= max(1, total)
    metrics["abstain_accuracy"] /= max(1, total)
    metrics["abstain_precision"] = abstain_tp / max(1, abstain_pred)
    metrics["abstain_recall"] = abstain_tp / max(1, abstain_gold)
    metrics["raw_recall_at_k"] /= max(1, total)
    metrics["latest_slice_recall"] = latest_slice_hits / max(1, latest_slice_total)
    metrics["latest_slice_conflict_accuracy"] = latest_conflict_hits / max(1, latest_conflict_total)
    metrics["false_merge_rate"] = false_merges / max(1, branch_positive)
    metrics["unnecessary_branch_rate"] = unnecessary_branches / max(1, branch_negative)
    metrics["header_fake_fact_rate"] = fake_headers / max(1, mature_header_count)
    metrics["avg_header_value_purity"] = purity_sum / max(1, mature_header_count)
    metrics["avg_headers_used"] /= max(1, total)
    metrics["avg_retrieval_latency_ms"] /= max(1, total)
    metrics["mature_header_count"] = mature_header_count
    metrics["samples_with_mature_headers"] = mature_header_samples
    metrics["abstain_tp"] = abstain_tp
    metrics["abstain_fp"] = abstain_fp
    metrics["abstain_tn"] = abstain_tn
    metrics["abstain_fn"] = abstain_fn
    if collect_debug:
        metrics["regime_mismatch"] = debug["failure_tags"]["regime_mismatch"]
        metrics["time_bucket_mismatch"] = debug["failure_tags"]["time_bucket_mismatch"]
        metrics["attribute_mismatch"] = debug["failure_tags"]["attribute_mismatch"]
        metrics["wrong_answer_should_abstain"] = debug["failure_buckets"]["wrong_answer_should_abstain"]
        metrics["wrong_answer_should_answer"] = debug["failure_buckets"]["wrong_answer_should_answer"]
        metrics["false_abstain"] = debug["failure_buckets"]["false_abstain"]
        metrics["debug"] = debug
    else:
        metrics["regime_mismatch"] = 0
        metrics["time_bucket_mismatch"] = 0
        metrics["attribute_mismatch"] = 0
        metrics["wrong_answer_should_abstain"] = 0
        metrics["wrong_answer_should_answer"] = 0
        metrics["false_abstain"] = 0
    return metrics


def ensure_results_header(path: str):
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    header_line = "\t".join(RESULTS_HEADER)
    if not os.path.exists(path):
        Path(path).write_text(header_line + "\n")
        return

    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle]

    if not lines:
        Path(path).write_text(header_line + "\n")
        return

    if lines[0] == header_line:
        return

    old_header = lines[0].split("\t")
    migrated = [header_line]
    for line in lines[1:]:
        if not line:
            continue
        values = line.split("\t")
        row_map = {key: values[idx] for idx, key in enumerate(old_header) if idx < len(values)}
        migrated.append("\t".join(row_map.get(key, "") for key in RESULTS_HEADER))

    Path(path).write_text("\n".join(migrated) + "\n")


def format_results_value(key: str, value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}" if "latency" not in key and "headers_used" not in key else f"{value:.3f}"
    return tsv_safe(str(value))


def append_results_row(path: str, row: dict):
    ensure_results_header(path)
    values = [format_results_value(key, row.get(key, "")) for key in RESULTS_HEADER]
    with open(path, "a") as handle:
        handle.write("\t".join(values) + "\n")


def default_dataset_path():
    return os.path.join(tempfile.gettempdir(), "ucmd_v36_synth.jsonl")


def run_experiment():
    if not cfg.dataset_path and os.getenv("UCMD_WRITE_DATA", "0") == "1":
        cfg.dataset_path = default_dataset_path()

    samples = build_samples(cfg)
    ds = SynthDataset(samples)
    codec = FixedFeatureCodec(ds, cfg.d_field)
    model = UCMDv31(codec, ds)
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.wd)
    mx.eval(model.parameters())

    indices = list(range(len(ds)))
    split = int(0.9 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_data = [ds[idx] for idx in train_idx]
    test_data = [ds[idx] for idx in test_idx]
    hard_eval_samples = build_hard_eval_samples(cfg)
    hard_eval_data = [ds.encode_sample(sample) for sample in hard_eval_samples]
    train_abstain_rate = sum(sample["abstain"] for sample in train_data) / max(1, len(train_data))
    test_abstain_rate = sum(sample["abstain"] for sample in test_data) / max(1, len(test_data))

    loss_and_grad = nn.value_and_grad(model, batch_loss)

    print(f"samples: {len(ds)} | train: {len(train_data)} | test: {len(test_data)}")
    print(f"hard eval samples: {len(hard_eval_data)}")
    print(f"d_model: {cfg.d_model} | h_max: {cfg.h_max} | n_hot: {cfg.n_hot}")
    print(f"train abstain rate: {train_abstain_rate:.3f} | test abstain rate: {test_abstain_rate:.3f}")
    print(f"results log: {cfg.results_path}")
    print("training only the small MLP modules and heads; field features are fixed")

    train_probe = train_data[: min(cfg.batch_size, len(train_data))]
    encountered_non_finite = False
    for epoch in range(cfg.epochs):
        running = 0.0
        steps = 0
        for batch_indices in batch_iter(train_idx, cfg.batch_size, shuffle=True):
            batch = [ds[idx] for idx in batch_indices]
            loss, grads = loss_and_grad(model, batch)
            loss_value = scalar(loss)
            grad_norm = grad_global_norm(grads)
            grad_norm_value = scalar(grad_norm)

            if not math.isfinite(loss_value) or not math.isfinite(grad_norm_value):
                print("non-finite loss encountered; stopping training early")
                print(f"loss: {loss_value} | grad_norm: {grad_norm_value}")
                debug_parts = batch_loss_parts(model, batch)
                print(
                    "loss parts: "
                    f"answer={scalar(debug_parts['answer_loss']):.4f} "
                    f"abstain={scalar(debug_parts['abstain_loss']):.4f} "
                    f"branch={scalar(debug_parts['branch_loss']):.4f} "
                    f"scope={scalar(debug_parts['scope_loss']):.4f} "
                    f"retrieval={scalar(debug_parts['retrieval_loss']):.4f}"
                )
                encountered_non_finite = True
                break

            grads, _ = clip_grad_norm(grads, cfg.grad_clip_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            running += loss_value
            steps += 1

        if encountered_non_finite:
            break

        metrics = evaluate(model, test_data[: min(128, len(test_data))])
        probe_parts = batch_loss_parts(model, train_probe)
        print(
            f"epoch {epoch + 1:2d} | train loss {running / max(1, steps):.4f} "
            f"| joint acc {metrics['joint_answer_or_abstain_acc']:.3f} "
            f"| abstain acc {metrics['abstain_accuracy']:.3f} "
            f"| purity {metrics['avg_header_value_purity']:.3f} "
            f"| answer {scalar(probe_parts['answer_loss']):.4f} "
            f"| abstain {scalar(probe_parts['abstain_loss']):.4f} "
            f"| branch {scalar(probe_parts['branch_loss']):.4f} "
            f"| scope {scalar(probe_parts['scope_loss']):.4f} "
            f"| retrieval {scalar(probe_parts['retrieval_loss']):.4f}"
        )

    if encountered_non_finite:
        return {
            "config": {
                "samples": cfg.n_samples,
                "epochs": cfg.epochs,
                "k_rerank": cfg.k_rerank,
                "results_path": cfg.results_path,
            },
            "non_finite": True,
        }

    final_metrics = evaluate(
        model,
        test_data,
        collect_debug=True,
        max_failure_examples=cfg.debug_failure_examples,
    )

    print("\n--- summary ---")
    print(f"joint_answer_or_abstain_acc: {final_metrics['joint_answer_or_abstain_acc']:.4f}")
    print(f"abstain_accuracy:           {final_metrics['abstain_accuracy']:.4f}")
    print(f"abstain_precision:          {final_metrics['abstain_precision']:.4f}")
    print(f"abstain_recall:             {final_metrics['abstain_recall']:.4f}")
    print(
        "abstain_confusion:         "
        f"tp={final_metrics['abstain_tp']} "
        f"fp={final_metrics['abstain_fp']} "
        f"tn={final_metrics['abstain_tn']} "
        f"fn={final_metrics['abstain_fn']}"
    )
    print(f"raw_recall_at_{cfg.k_rerank}:            {final_metrics['raw_recall_at_k']:.4f}")
    print(f"latest_slice_recall:        {final_metrics['latest_slice_recall']:.4f}")
    print(f"latest_slice_conflict_acc:  {final_metrics['latest_slice_conflict_accuracy']:.4f}")
    print(f"false_merge_rate:           {final_metrics['false_merge_rate']:.4f}")
    print(f"unnecessary_branch_rate:    {final_metrics['unnecessary_branch_rate']:.4f}")
    print(f"header_fake_fact_rate:      {final_metrics['header_fake_fact_rate']:.4f}")
    print(f"avg_header_value_purity:    {final_metrics['avg_header_value_purity']:.4f}")
    print(f"mature_header_count:        {final_metrics['mature_header_count']}")
    print(f"samples_with_mature_headers:{final_metrics['samples_with_mature_headers']}")
    print(f"avg_headers_used:           {final_metrics['avg_headers_used']:.2f}")
    print(f"avg_retrieval_latency_ms:   {final_metrics['avg_retrieval_latency_ms']:.2f}")
    print_eval_debug(final_metrics)

    if hard_eval_data:
        hard_metrics = evaluate(
            model,
            hard_eval_data,
            collect_debug=True,
            max_failure_examples=cfg.debug_failure_examples,
        )
        print("\n--- hard eval ---")
        print(f"joint_answer_or_abstain_acc: {hard_metrics['joint_answer_or_abstain_acc']:.4f}")
        print(f"abstain_accuracy:           {hard_metrics['abstain_accuracy']:.4f}")
        print(f"abstain_precision:          {hard_metrics['abstain_precision']:.4f}")
        print(f"abstain_recall:             {hard_metrics['abstain_recall']:.4f}")
        print(
            "abstain_confusion:         "
            f"tp={hard_metrics['abstain_tp']} "
            f"fp={hard_metrics['abstain_fp']} "
            f"tn={hard_metrics['abstain_tn']} "
            f"fn={hard_metrics['abstain_fn']}"
        )
        print(f"raw_recall_at_{cfg.k_rerank}:            {hard_metrics['raw_recall_at_k']:.4f}")
        print(f"latest_slice_recall:        {hard_metrics['latest_slice_recall']:.4f}")
        print(f"latest_slice_conflict_acc:  {hard_metrics['latest_slice_conflict_accuracy']:.4f}")
        print(f"false_merge_rate:           {hard_metrics['false_merge_rate']:.4f}")
        print(f"unnecessary_branch_rate:    {hard_metrics['unnecessary_branch_rate']:.4f}")
        print(f"header_fake_fact_rate:      {hard_metrics['header_fake_fact_rate']:.4f}")
        print(f"avg_header_value_purity:    {hard_metrics['avg_header_value_purity']:.4f}")
        print(f"mature_header_count:        {hard_metrics['mature_header_count']}")
        print(f"samples_with_mature_headers:{hard_metrics['samples_with_mature_headers']}")
        print(f"avg_headers_used:           {hard_metrics['avg_headers_used']:.2f}")
        print(f"avg_retrieval_latency_ms:   {hard_metrics['avg_retrieval_latency_ms']:.2f}")
        print_eval_debug(hard_metrics)
    else:
        hard_metrics = None

    mem = Memory(model, ds)
    sample = test_data[0]
    for event in sample["events"]:
        mem.add_observation(event["raw"], provenance=event["raw"]["provenance"])
    result = mem.retrieve(sample["query"]["raw"])

    gold = "ABSTAIN" if sample["abstain"] else ds.values[sample["answer"]]
    pred = "ABSTAIN" if result["abstain"] else result["answer"]

    print("\n--- demo ---")
    print("query:", sample["query"]["raw"])
    print("gold:", gold)
    print("pred:", pred)
    print("abstain_prob:", round(result["abstain_prob"], 4))
    print("evidence_strength:", round(result["evidence_strength"], 4))
    print("mass_gap:", round(result["mass_gap"], 4))
    print("conflict_mass:", round(result["conflict_mass"], 4))
    print("top_time_penalty:", round(result["top_time_penalty"], 4))
    print("headers used:", len(result["headers"]))
    print("raw evidence returned:", len(result["raw"]))

    if result["abstain"]:
        action = mem.decide_after_abstain(sample["query"]["raw"], result, high_risk=False)
        print("after abstain:", action)

    return {
        "config": {
            "samples": cfg.n_samples,
            "epochs": cfg.epochs,
            "k_rerank": cfg.k_rerank,
            "results_path": cfg.results_path,
        },
        "non_finite": False,
        "final_metrics": final_metrics,
        "hard_metrics": hard_metrics,
        "demo": {
            "query": sample["query"]["raw"],
            "gold": gold,
            "pred": pred,
            "abstain_prob": scalar(result["abstain_prob"]),
            "evidence_strength": result["evidence_strength"],
            "mass_gap": result["mass_gap"],
            "conflict_mass": result["conflict_mass"],
            "top_time_penalty": result["top_time_penalty"],
        },
    }


def main():
    run_experiment()


if __name__ == "__main__":
    main()
