from __future__ import annotations

import copy
import json


SEED_CANDIDATE = {
    "mode_router": "balanced_temporal",
    "current_strategy": "latest_with_support",
    "temporal_strategy": "explicit_or_session_time",
    "multi_hop_strategy": "aggregate_two_hops",
    "answer_style": "mode_default",
    "abstain_profile": "answerable_friendly",
    "generic_answer_rule": "reject_full_sentence",
    "explanation_policy": "brief_abstain_reason",
}


SMOKE_RUN_CONFIG = {
    "eval_budget": 96,
    "max_proposals": 3,
    "max_reflective_examples_per_component": 12,
    "max_explanation_audit_examples": 12,
}


def candidate_to_json(candidate: dict[str, str]) -> str:
    return json.dumps(candidate, indent=2, ensure_ascii=True, sort_keys=True)


def clone_candidate(candidate: dict[str, str]) -> dict[str, str]:
    return copy.deepcopy(candidate)
