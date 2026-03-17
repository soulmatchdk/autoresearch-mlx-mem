from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


QUERY_MODES = ("current", "temporal", "multi_hop", "abstain_like")
CONFIDENCE_BANDS = ("low", "medium", "high")


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_mode": self.query_mode,
            "answer_candidate": self.answer_candidate,
            "should_abstain": self.should_abstain,
            "explanation": self.explanation,
            "confidence_band": self.confidence_band,
            "support_items": [item.to_dict() for item in self.support_items],
            "answer_style": self.answer_style,
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
        "query_mode_rubric",
        "mode_routing_bias",
        "current_policy",
        "temporal_policy",
        "temporal_grounding_rule",
        "multi_hop_policy",
        "multi_hop_evidence_requirement",
        "abstain_policy",
        "abstain_guardrail_answerable",
        "generic_answer_rejection_rule",
        "answer_style_policy",
        "answer_synthesis_policy",
        "confidence_policy",
        "explanation_policy",
    )
    missing = [name for name in required if not isinstance(candidate.get(name), str) or not candidate[name].strip()]
    if missing:
        raise ValueError(f"candidate is missing required text components: {', '.join(missing)}")
    return candidate


def validate_query_mode(value: str) -> str:
    if value not in QUERY_MODES:
        raise ValueError(f"invalid query mode: {value}")
    return value


def validate_confidence_band(value: str) -> str:
    if value not in CONFIDENCE_BANDS:
        raise ValueError(f"invalid confidence band: {value}")
    return value
