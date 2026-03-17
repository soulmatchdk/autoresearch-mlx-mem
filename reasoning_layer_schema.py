from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


QUERY_MODES = ("current", "temporal", "multi_hop", "abstain_like")
CONFIDENCE_BANDS = ("low", "medium", "high")
MODE_ROUTERS = ("balanced_temporal", "temporal_friendly", "multi_hop_friendly", "abstain_first")
CURRENT_STRATEGIES = ("latest_only", "latest_with_support")
TEMPORAL_STRATEGIES = ("latest_with_time_anchor", "explicit_or_session_time", "ordered_history")
MULTI_HOP_STRATEGIES = ("aggregate_two_hops", "aggregate_three_hops", "abstain_first")
ANSWER_STYLE_POLICIES = ("mode_default", "short_slot_value", "short_date", "short_entity", "abstain")
ABSTAIN_PROFILES = ("strict", "balanced", "answerable_friendly")
GENERIC_ANSWER_RULES = ("allow_compact_span", "reject_full_sentence", "reject_long_or_generic")
EXPLANATION_POLICIES = ("brief_grounded", "brief_abstain_reason")


@dataclass
class MemoryEvidence:
    kind: str
    score: float
    text: str
    value: str | None
    entity: str | None
    attribute: str | None
    regime: str | None
    session_idx: int
    session_time: str | None
    dia_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryContext:
    query_id: str
    question: str
    entity: str | None
    attribute: str | None
    regime: str | None
    evidence_items: list[MemoryEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "entity": self.entity,
            "attribute": self.attribute,
            "regime": self.regime,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
        }


@dataclass
class ReasoningOutput:
    query_mode: str
    answer_candidate: str | None
    should_abstain: bool
    explanation: str
    confidence_band: str
    support_items: list[MemoryEvidence] = field(default_factory=list)
    answer_style: str = "single"
    evidence_strategy: str = ""
    abstain_profile: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_mode": self.query_mode,
            "answer_candidate": self.answer_candidate,
            "should_abstain": self.should_abstain,
            "explanation": self.explanation,
            "confidence_band": self.confidence_band,
            "support_items": [item.to_dict() for item in self.support_items],
            "answer_style": self.answer_style,
            "evidence_strategy": self.evidence_strategy,
            "abstain_profile": self.abstain_profile,
        }


@dataclass
class EvaluationBatch:
    outputs: list[dict[str, Any]]
    scores: list[float]
    trajectories: list[dict[str, Any]]
    objective_scores: list[dict[str, float]]
    aggregate_metrics: dict[str, Any]


def validate_candidate(candidate: dict[str, str]) -> dict[str, str]:
    required = (
        "mode_router",
        "current_strategy",
        "temporal_strategy",
        "multi_hop_strategy",
        "answer_style",
        "abstain_profile",
        "generic_answer_rule",
        "explanation_policy",
    )
    missing = [name for name in required if not isinstance(candidate.get(name), str) or not candidate[name].strip()]
    if missing:
        raise ValueError(f"candidate is missing required text components: {', '.join(missing)}")
    allowed = {
        "mode_router": MODE_ROUTERS,
        "current_strategy": CURRENT_STRATEGIES,
        "temporal_strategy": TEMPORAL_STRATEGIES,
        "multi_hop_strategy": MULTI_HOP_STRATEGIES,
        "answer_style": ANSWER_STYLE_POLICIES,
        "abstain_profile": ABSTAIN_PROFILES,
        "generic_answer_rule": GENERIC_ANSWER_RULES,
        "explanation_policy": EXPLANATION_POLICIES,
    }
    invalid = []
    for key, values in allowed.items():
        value = candidate.get(key, "").strip().lower().replace("-", "_")
        if value in values:
            continue
        if any(option in value for option in values):
            continue
        if candidate.get(key) not in values:
            invalid.append(f"{key}={candidate.get(key)!r}")
    if invalid:
        raise ValueError(f"candidate has invalid discrete choices: {', '.join(invalid)}")
    return candidate


def validate_query_mode(value: str) -> str:
    if value not in QUERY_MODES:
        raise ValueError(f"invalid query mode: {value}")
    return value


def validate_confidence_band(value: str) -> str:
    if value not in CONFIDENCE_BANDS:
        raise ValueError(f"invalid confidence band: {value}")
    return value
