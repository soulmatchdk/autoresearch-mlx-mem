from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from locomo_eval import events_by_conversation, headers_by_conversation, load_jsonl, load_metadata
from locomo_reasoning_eval import evaluate_reasoning_batch, sample_queries
from reasoning_layer_schema import validate_candidate


class ReasoningGEPAAdapter(GEPAAdapter[dict[str, Any], dict[str, Any], dict[str, Any]]):
    """Official GEPA adapter backed by the frozen LoCoMo reasoning evaluator."""

    def __init__(
        self,
        events_path: str = "locomo_adapted/events.jsonl",
        headers_path: str = "locomo_adapted/headers.jsonl",
        queries_path: str = "locomo_adapted/queries.jsonl",
        metadata_path: str = "locomo_adapted/metadata.json",
        seed: int = 13,
    ):
        self.events_path = str(events_path)
        self.headers_path = str(headers_path)
        self.queries_path = str(queries_path)
        self.metadata_path = str(metadata_path)
        self.seed = seed
        self.events = load_jsonl(Path(self.events_path))
        self.headers = load_jsonl(Path(self.headers_path))
        self.queries = load_jsonl(Path(self.queries_path))
        self.metadata = load_metadata(Path(self.metadata_path)) if Path(self.metadata_path).exists() else {}
        self.conversations = events_by_conversation(self.events)
        self.headers_by_conv = headers_by_conversation(self.headers)
        self._batch_summaries: dict[int, dict[str, Any]] = {}

    def sample_batch(self, budget: int, seed: int | None = None) -> list[dict[str, Any]]:
        return sample_queries(self.queries, budget, self.seed if seed is None else seed)

    def evaluate(self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        validate_candidate(candidate)
        local_batch, summary = evaluate_reasoning_batch(
            candidate=candidate,
            query_batch=batch,
            conversations=self.conversations,
            headers_by_conv=self.headers_by_conv,
            metadata=self.metadata,
            capture_traces=True,
        )
        eval_batch = EvaluationBatch(
            outputs=local_batch.outputs,
            scores=local_batch.scores,
            trajectories=local_batch.trajectories if capture_traces else None,
            objective_scores=local_batch.objective_scores,
        )
        self._batch_summaries[id(eval_batch)] = summary
        return eval_batch

    def evaluate_candidate(
        self,
        candidate: dict[str, str],
        budget: int,
        config: dict[str, Any] | None = None,
        capture_traces: bool = False,
    ) -> tuple[EvaluationBatch, dict[str, Any]]:
        config = config or {}
        batch = self.sample_batch(budget=budget, seed=config.get("seed", self.seed))
        eval_batch = self.evaluate(batch=batch, candidate=candidate, capture_traces=capture_traces)
        return eval_batch, self.summary_for(eval_batch)

    def summary_for(self, eval_batch: EvaluationBatch) -> dict[str, Any]:
        return self._batch_summaries.get(id(eval_batch), {})

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        validate_candidate(candidate)
        allowed = set(components_to_update or candidate.keys())
        dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trajectory in eval_batch.trajectories or []:
            targets = self._reflection_targets(trajectory)
            for component in targets:
                if component not in allowed:
                    continue
                feedback = dict(trajectory["Feedback"])
                feedback["reflection_target"] = component
                record = {
                    "component": component,
                    "query_id": trajectory["query_id"],
                    "Inputs": dict(trajectory["Inputs"]),
                    "Generated Outputs": dict(trajectory["Generated Outputs"]),
                    "Feedback": feedback,
                }
                dataset[component].append(record)
        return dict(dataset)

    def _reflection_targets(self, trajectory: dict[str, Any]) -> list[str]:
        feedback = trajectory["Feedback"]
        generated = trajectory["Generated Outputs"]
        targets = []
        gold_mode = feedback.get("gold_mode_label")
        predicted_mode = generated.get("query_mode")
        failure_bucket = feedback.get("failure_bucket")
        if gold_mode and predicted_mode and gold_mode != predicted_mode:
            targets.append("query_mode_rubric")
        if predicted_mode == "current":
            targets.append("current_policy")
        elif predicted_mode == "temporal":
            targets.append("temporal_policy")
        elif predicted_mode == "multi_hop":
            targets.append("multi_hop_policy")
        else:
            targets.append("query_mode_rubric")
        if failure_bucket in {"false_abstain", "false_confident_answer"}:
            targets.append("abstain_policy")
        if feedback.get("explanation_quality") != "good":
            targets.append("explanation_policy")
        if generated.get("answer") is not None or failure_bucket == "false_confident_answer":
            targets.append("answer_synthesis_policy")
        seen = set()
        out = []
        for target in targets:
            if target in seen:
                continue
            seen.add(target)
            out.append(target)
        return out
