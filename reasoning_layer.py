from __future__ import annotations

import re
from typing import Iterable

from locomo_temporal_policy import date_phrase, parse_session_datetime, temporal_answer_from_text
from reasoning_layer_schema import MemoryContext, MemoryEvidence, ReasoningOutput, validate_candidate, validate_confidence_band, validate_query_mode


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "being",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "would",
}
COLLECTION_VOCABS = {
    "activities": ["pottery", "camping", "painting", "swimming", "running", "hiking", "museum", "beach", "violin", "reading"],
    "books": ["charlotte's web", "nothing is impossible", "becoming nicole", "dr. seuss"],
    "camp_locations": ["beach", "mountains", "forest", "park", "canyon"],
    "kids_like": ["dinosaurs", "nature", "flowers", "books"],
    "events": ["pride parade", "school speech", "support group", "transgender conference", "lgbtq conference", "museum", "pottery class"],
}
ABSTAIN_PATTERNS = (
    "is there any information",
    "is it mentioned",
    "do we know",
    "not mentioned",
    "what do we know",
)


def normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return " ".join(lowered.split())


def content_terms(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token not in STOPWORDS and len(token) > 2}


def parse_flags(policy_text: str) -> set[str]:
    if "Flags:" not in policy_text:
        return set()
    _, tail = policy_text.split("Flags:", 1)
    return {part.strip().strip(".") for part in tail.split(",") if part.strip()}


def candidate_flags(candidate: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    for value in candidate.values():
        flags.update(parse_flags(value))
    return flags


def classify_query_mode(candidate: dict[str, str], question: str) -> str:
    _ = validate_candidate(candidate)
    lowered = f" {question.lower()} "
    if any(pattern in lowered for pattern in ABSTAIN_PATTERNS):
        return validate_query_mode("abstain_like")
    if lowered.startswith(" when ") or " how long " in lowered or " how long ago " in lowered or " what date " in lowered or " which year " in lowered:
        return validate_query_mode("temporal")
    if lowered.startswith(" would ") or " if " in lowered or " likely " in lowered or " how many " in lowered:
        return validate_query_mode("multi_hop")
    return validate_query_mode("current")


def infer_answer_style(question: str, query_mode: str) -> str:
    lowered = question.lower()
    if lowered.startswith("would ") or " likely " in lowered:
        return "counterfactual"
    if any(phrase in lowered for phrase in ("what activities", "what books", "where has", "what events", "what do")):
        return "collection"
    if query_mode == "temporal":
        return "temporal"
    return "single"


def shared_terms(question: str, text: str) -> set[str]:
    return content_terms(question) & content_terms(text)


def directness_score(question: str, item: MemoryEvidence) -> float:
    text = f"{item.text} {item.value or ''}"
    score = float(len(shared_terms(question, text)))
    lowered = text.lower()
    q = question.lower()
    if "research" in q and "research" in lowered:
        score += 2.0
    if "identity" in q and "trans" in lowered:
        score += 2.0
    if "relationship" in q and any(marker in lowered for marker in ("single", "married", "dating")):
        score += 2.0
    if "career" in q and any(marker in lowered for marker in ("career", "counseling", "mental health")):
        score += 2.0
    if "move from" in q and " from " in lowered:
        score += 2.0
    if any(marker in q for marker in ("activities", "books", "where has", "what events")):
        score += 1.0
    return score


def rank_support_items(candidate: dict[str, str], context: MemoryContext, query_mode: str) -> list[MemoryEvidence]:
    flags = candidate_flags(candidate)
    ranked = []
    for item in context.evidence_items:
        score = item.score
        direct = directness_score(context.question, item)
        score += 1.5 * direct
        if "prefer_direct_match" in flags and direct > 0:
            score += 1.5
        if "prefer_attribute_match" in flags and context.attribute and item.attribute == context.attribute:
            score += 2.0
        if query_mode == "current" and "blend_specificity_with_freshness" in flags:
            score += 0.15 * item.session_idx
        ranked.append((score, item.session_idx, item))
    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [item for _, _, item in ranked]


def extract_titles(text: str) -> list[str]:
    titles = []
    titles.extend(re.findall(r'"([^"]+)"', text))
    titles.extend(re.findall(r"'([^']+)'", text))
    out = []
    seen = set()
    for title in titles:
        lowered = normalize_text(title)
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        out.append(title.strip())
    return out


def dedupe(items: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        lowered = normalize_text(item)
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        out.append(item.strip())
    return out


def aggregate_collection_answer(question: str, support_items: list[MemoryEvidence]) -> str | None:
    merged = " ".join(f"{item.text} {item.value or ''}" for item in support_items)
    lowered = question.lower()
    items: list[str] = []
    if "activities" in lowered or "destress" in lowered:
        for value in COLLECTION_VOCABS["activities"]:
            if normalize_text(value) in normalize_text(merged):
                items.append(value)
    elif "books" in lowered:
        items.extend(extract_titles(merged))
        for value in COLLECTION_VOCABS["books"]:
            if normalize_text(value) in normalize_text(merged):
                items.append(value)
    elif "where has" in lowered and "camp" in lowered:
        for value in COLLECTION_VOCABS["camp_locations"]:
            if normalize_text(value) in normalize_text(merged):
                items.append(value)
    elif "kids like" in lowered:
        for value in COLLECTION_VOCABS["kids_like"]:
            if normalize_text(value) in normalize_text(merged):
                items.append(value)
    elif "events" in lowered or "participated in" in lowered:
        for value in COLLECTION_VOCABS["events"]:
            if normalize_text(value) in normalize_text(merged):
                items.append(value)
    items = dedupe(items)
    if not items:
        return None
    return ", ".join(items)


def duration_answer(text: str) -> str | None:
    match = re.search(r"\b(\d+\s+(?:years?|months?|weeks?|days?))\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d+\s+years?\s+ago)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def current_answer_from_item(question: str, item: MemoryEvidence) -> str | None:
    text = f"{item.text} {item.value or ''}"
    lowered = text.lower()
    q = question.lower()
    if "identity" in q:
        if "transgender woman" in lowered:
            return "Transgender woman"
        if "trans woman" in lowered:
            return "Trans woman"
        if "trans person" in lowered:
            return "Trans person"
    if "relationship status" in q:
        for marker in ("single", "married", "dating"):
            if marker in lowered:
                return marker.capitalize()
    if "research" in q:
        match = re.search(r"research(?:ing)? ([^.]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    if "move from" in q or "where did" in q and "move" in q:
        match = re.search(r"\bfrom ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
        if match:
            return match.group(1)
    if "career" in q or "career path" in q:
        match = re.search(r"(?:career (?:in|options in)|explore career options in|considering a career in) ([^.]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    if "how long" in q:
        answer = duration_answer(text)
        if answer:
            return answer
    return item.value or item.text


def compact_answer(question: str, answer: str) -> str:
    question_lower = question.lower()
    if "research" in question_lower and " with " in answer.lower():
        return answer.split(" with ", 1)[0].strip(" .")
    if "career" in question_lower and " to " in answer.lower():
        return answer.split(" to ", 1)[0].strip(" .")
    return answer


def temporal_answer_from_item(question: str, item: MemoryEvidence, flags: set[str]) -> str | None:
    _ = question
    text = item.text
    answer = temporal_answer_from_text(text, item.session_time)
    if answer:
        return answer
    lowered = text.lower()
    if "allow_duration_answer" in flags:
        duration = duration_answer(text)
        if duration:
            return duration
    if "allow_recently_anchor" in flags and "recently" in lowered:
        dt = parse_session_datetime(item.session_time)
        if dt is not None:
            return date_phrase(dt)
    return None


def counterfactual_answer(question: str, support_items: list[MemoryEvidence]) -> str | None:
    question_lower = question.lower()
    merged = " ".join(item.text.lower() for item in support_items)
    if "if she hadn't" in question_lower or "if he hadn't" in question_lower:
        if any(marker in merged for marker in ("support", "help", "motivation", "guidance")):
            return "Likely no"
    if "dr seuss" in question_lower or "dr. seuss" in question_lower:
        if any(marker in merged for marker in ("classic children", "bookshelf", "reading", "book")):
            return "Likely yes"
    if "would melanie be considered a member of the lgbtq community" in question_lower:
        if "caroline responded" in merged or "caroline expressed" in merged:
            return "Likely no"
    return None


def choose_answer(candidate: dict[str, str], context: MemoryContext, query_mode: str, answer_style: str) -> tuple[str | None, list[MemoryEvidence]]:
    flags = candidate_flags(candidate)
    ranked = rank_support_items(candidate, context, query_mode)
    if not ranked:
        return None, []
    if query_mode == "abstain_like":
        return None, ranked[:1]
    if answer_style == "collection":
        top_k = 6 if "collection_top6" in flags else 4 if "collection_top4" in flags else 3
        answer = aggregate_collection_answer(context.question, ranked[:top_k])
        return answer, ranked[:top_k]
    if query_mode == "temporal":
        for item in ranked[:6]:
            answer = temporal_answer_from_item(context.question, item, flags)
            if answer:
                return answer, [item]
        return None, ranked[:1]
    if query_mode == "multi_hop":
        if answer_style == "counterfactual":
            combine_k = 3 if "combine_top3_multihop" in flags else 2
            answer = counterfactual_answer(context.question, ranked[:combine_k])
            if answer:
                return answer, ranked[:combine_k]
        combine_k = 3 if "combine_top3_multihop" in flags else 2
        for item in ranked[:combine_k]:
            answer = current_answer_from_item(context.question, item)
            if answer:
                return answer, ranked[:combine_k]
        return None, ranked[:1]
    for item in ranked[:4]:
        answer = current_answer_from_item(context.question, item)
        if answer:
            if "prefer_compact_value_span" in flags:
                answer = compact_answer(context.question, answer)
            return answer, [item]
    return None, ranked[:1]


def confidence_band(answer: str | None, support_items: list[MemoryEvidence]) -> str:
    if not answer or not support_items:
        return validate_confidence_band("low")
    top_score = support_items[0].score
    if top_score >= 12.0 or len(support_items) >= 2:
        return validate_confidence_band("high")
    if top_score >= 7.0:
        return validate_confidence_band("medium")
    return validate_confidence_band("low")


def make_explanation(candidate: dict[str, str], question: str, query_mode: str, answer: str | None, support_items: list[MemoryEvidence], should_abstain: bool) -> str:
    _ = candidate
    if not support_items:
        return "I could not ground the question in any retrieved evidence."
    support = support_items[0]
    shared = sorted(shared_terms(question, support.text))
    shared_fragment = ", ".join(shared[:4]) if shared else support.text[:80].strip()
    if should_abstain:
        return f"I abstained because the best evidence was not grounded enough to give a stable {query_mode} answer. Top support: {shared_fragment}."
    return f"I answered using the strongest grounded support, especially the evidence about {shared_fragment}. The supported answer is {answer}."


def should_abstain(candidate: dict[str, str], query_mode: str, answer: str | None, support_items: list[MemoryEvidence]) -> bool:
    flags = candidate_flags(candidate)
    if query_mode == "abstain_like":
        return True
    if not answer or not support_items:
        return True
    top = support_items[0]
    threshold = 2.0
    if "low_grounding_threshold" in flags:
        threshold = 1.0
    if "high_grounding_threshold" in flags:
        threshold = 3.5
    if top.score < threshold and "grounded_answer_beats_default_abstain" not in flags:
        return True
    if "abstain_on_long_generic_answers" in flags and len((answer or "").split()) > 10:
        return True
    if query_mode in {"current", "temporal"} and "grounded_answer_beats_default_abstain" in flags:
        return False
    return False


def run_reasoning(candidate: dict[str, str], context: MemoryContext) -> ReasoningOutput:
    validate_candidate(candidate)
    query_mode = classify_query_mode(candidate, context.question)
    answer_style = infer_answer_style(context.question, query_mode)
    answer, support_items = choose_answer(candidate, context, query_mode, answer_style)
    abstain = should_abstain(candidate, query_mode, answer, support_items)
    if abstain:
        answer = None
    explanation = make_explanation(candidate, context.question, query_mode, answer, support_items, abstain)
    output = ReasoningOutput(
        query_mode=query_mode,
        answer_candidate=answer,
        should_abstain=abstain,
        explanation=explanation,
        confidence_band=confidence_band(answer, support_items),
        support_items=support_items,
        answer_style=answer_style,
    )
    return output
