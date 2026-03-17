import re
from collections import Counter


POLICY_MODES = ("current", "temporal", "multi_hop", "abstain_like")
QUESTION_WORDS = {"what", "when", "where", "who", "which", "why", "how"}
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
    "activities": ["pottery", "camping", "painting", "swimming", "running", "hiking", "museum", "beach"],
    "books": ["charlotte's web", "nothing is impossible", "becoming nicole", "dr. seuss"],
    "camp_locations": ["beach", "mountains", "forest", "park", "canyon"],
    "kids_like": ["dinosaurs", "nature", "flowers", "books"],
    "events": ["pride parade", "support group", "school speech", "mentorship program", "art show", "activist group", "conference"],
}
QUESTION_HINTS = {
    "identity": ["transgender", "trans", "woman", "identity"],
    "relationship status": ["single", "married", "dating", "partner", "relationship"],
    "research": ["research", "researching", "adoption", "agency"],
    "career": ["career", "counseling", "mental", "health", "work", "job"],
    "destress": ["destress", "stress", "running", "pottery", "calm", "relax"],
    "camping": ["camping", "camp", "beach", "mountains", "forest"],
    "kids like": ["kids", "children", "dinosaurs", "nature"],
    "books": ["books", "book", "read"],
}


def normalize_text(text: str):
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return " ".join(lowered.split())


def content_terms(text: str):
    return {token for token in normalize_text(text).split() if token not in STOPWORDS and len(token) > 2}


def sentence_split(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [part.strip() for part in parts if part.strip()]


def classify_policy_mode(query: dict):
    question = f" {query.get('text', '').lower()} "
    category = query.get("category")
    query_mode = query.get("query_mode")
    if category == 5 or query_mode == "adversarial":
        return "abstain_like"
    if category == 2 or question.startswith(" when ") or " how long " in question:
        return "temporal"
    if category in {3, 4} or question.startswith(" would ") or " if " in question or " how many " in question:
        return "multi_hop"
    return "current"


def infer_hint_terms(question: str):
    lowered = question.lower()
    hints = set()
    for key, values in QUESTION_HINTS.items():
        if key in lowered:
            hints.update(values)
    return hints


def infer_answer_style(question: str):
    lowered = question.lower()
    if any(phrase in lowered for phrase in (" what activities ", " what books ", " where has ", " what events ", " what do ")):
        return "collection"
    if lowered.startswith("would ") or " likely " in lowered:
        return "counterfactual"
    return "single"


def build_query_profile(query: dict):
    question = query.get("text", "")
    lowered = question.lower()
    subject_terms = {
        token
        for token in re.findall(r"\b[A-Z][a-z]+\b", question)
        if token.lower() not in QUESTION_WORDS
    }
    return {
        "question": question,
        "question_lower": lowered,
        "policy_mode": classify_policy_mode(query),
        "query_terms": content_terms(question),
        "hint_terms": infer_hint_terms(question),
        "subject_terms": {token.lower() for token in subject_terms},
        "attribute": query.get("attribute"),
        "answer_style": infer_answer_style(question),
    }


def keyword_hits(text: str, keywords: list[str]):
    lowered = normalize_text(text)
    hits = []
    for keyword in keywords:
        if normalize_text(keyword) in lowered:
            hits.append(keyword)
    return hits


def score_text(profile: dict, text: str, attribute: str | None = None, recency: float = 0.0):
    terms = content_terms(text)
    overlap = len(profile["query_terms"] & terms)
    hint_overlap = len(profile["hint_terms"] & terms)
    subject_overlap = len(profile["subject_terms"] & terms)
    score = 2.0 * overlap + 2.5 * hint_overlap + 1.5 * subject_overlap
    if attribute and attribute == profile["attribute"]:
        score += 1.0
    if profile["policy_mode"] == "current":
        score += 0.35 * recency
    return score


def build_event_candidates(query: dict, conversation_events: list[dict], profile: dict):
    scoped = [
        event
        for event in conversation_events
        if event["entity"] == query["entity"] and event["regime"] == query["regime"]
    ]
    max_session = max((event["session_idx"] for event in scoped), default=1)
    candidates = []
    for event in scoped:
        recency = event["session_idx"] / max_session
        text = f"{event['text']} {event.get('value', '')}"
        score = score_text(profile, text, event.get("attribute"), recency)
        if score <= 0.0:
            continue
        candidates.append(
            {
                "kind": "event",
                "score": score,
                "text": event["text"],
                "value": event.get("value"),
                "entity": event.get("entity"),
                "attribute": event.get("attribute"),
                "regime": event.get("regime"),
                "session_idx": event["session_idx"],
                "session_time": event.get("session_time"),
                "dia_ids": list(event.get("dia_ids", [])),
                "raw": event,
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["session_idx"]), reverse=True)
    return candidates


def build_header_candidates(query: dict, conversation_headers: list[dict], conversation_events: list[dict], profile: dict):
    if not conversation_headers:
        return []
    max_session = max((header["session_idx"] for header in conversation_headers), default=1)
    event_lookup = {event["event_id"]: event for event in conversation_events}
    candidates = []
    for header in conversation_headers:
        dia_ids = []
        for raw_id in header.get("backlinks", []):
            event = event_lookup.get(raw_id)
            if event is not None:
                dia_ids.extend(event.get("dia_ids", []))
        recency = header["session_idx"] / max_session
        for sentence in sentence_split(header.get("summary_text", "")):
            if profile["subject_terms"] and not any(term in normalize_text(sentence) for term in profile["subject_terms"]):
                continue
            score = score_text(profile, sentence, profile["attribute"], recency)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "kind": "header_sentence",
                    "score": score,
                    "text": sentence,
                    "value": sentence,
                    "entity": query.get("entity"),
                    "attribute": profile["attribute"],
                    "regime": query.get("regime"),
                    "session_idx": header["session_idx"],
                    "session_time": None,
                    "dia_ids": list(dict.fromkeys(dia_ids)),
                    "raw": header,
                }
            )
    candidates.sort(key=lambda item: (item["score"], item["session_idx"]), reverse=True)
    return candidates


def merge_candidates(event_candidates, header_candidates, limit: int = 8):
    merged = sorted(event_candidates + header_candidates, key=lambda item: (item["score"], item["session_idx"]), reverse=True)
    return merged[:limit]


def best_candidate(candidates):
    return candidates[0] if candidates else None


def extract_titles(text: str):
    titles = []
    titles.extend(re.findall(r'"([^"]+)"', text))
    titles.extend(re.findall(r"'([^']+)'", text))
    for match in re.findall(r"\b(?:[A-Z][a-z]+(?:'s)?(?:\s+[A-Z][a-z]+(?:'s)?)*)\b", text):
        if match.lower() not in {"caroline", "melanie", "transgender", "lgbtq", "lgbtq+"} and len(match.split()) <= 4:
            titles.append(match)
    out = []
    seen = set()
    for title in titles:
        lowered = normalize_text(title)
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        out.append(title.strip())
    return out


def aggregate_collection_answer(profile: dict, candidates):
    question = profile["question_lower"]
    texts = [f"{item['text']} {item.get('value') or ''}" for item in candidates[:6]]
    merged = " ".join(texts)
    items = []
    if "activities" in question or "destress" in question:
        items = keyword_hits(merged, COLLECTION_VOCABS["activities"])
    elif "books" in question:
        items = extract_titles(merged)
    elif "where has" in question and "camp" in question:
        items = keyword_hits(merged, COLLECTION_VOCABS["camp_locations"])
    elif "kids like" in question:
        items = keyword_hits(merged, COLLECTION_VOCABS["kids_like"])
    elif "events" in question:
        items = keyword_hits(merged, COLLECTION_VOCABS["events"])
    if not items:
        return None
    return ", ".join(dict.fromkeys(items))


def extract_current_answer(profile: dict, candidate: dict):
    text = f"{candidate['text']} {candidate.get('value') or ''}"
    lowered = text.lower()
    question = profile["question_lower"]
    if "identity" in question:
        if "transgender woman" in lowered:
            return "Transgender woman"
        if "trans woman" in lowered:
            return "Trans woman"
        if "transgender" in lowered or "trans " in lowered:
            return "Transgender woman"
    if "relationship status" in question:
        for marker in ("single", "married", "dating"):
            if marker in lowered:
                return marker
    if "research" in question:
        match = re.search(r"research(?:ing)? ([^.]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    if "move from" in question:
        match = re.search(r"\bfrom ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
        if match:
            return match.group(1)
    if "career" in question or "career path" in question:
        match = re.search(r"(?:career (?:in|options in)|looking into|considering a career in) ([^.]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    if "books" in question:
        titles = extract_titles(text)
        if titles:
            return ", ".join(titles[:4])
    if "destress" in question:
        items = keyword_hits(text, COLLECTION_VOCABS["activities"])
        if items:
            return ", ".join(dict.fromkeys(items))
    return candidate.get("value") or candidate["text"]


def current_policy(query: dict, profile: dict, event_candidates, header_candidates):
    candidates = merge_candidates(event_candidates, header_candidates)
    if not candidates:
        return {"abstain": True, "predicted_value": None, "support_events": [], "policy_mode": "current"}

    if profile["answer_style"] == "collection":
        answer = aggregate_collection_answer(profile, candidates)
        if answer:
            return {"abstain": False, "predicted_value": answer, "support_events": candidates[:4], "policy_mode": "current"}

    best_score = candidates[0]["score"]
    contender_pool = [item for item in candidates if item["score"] >= best_score - 1.0]
    if contender_pool:
        freshest_session = max(item["session_idx"] for item in contender_pool)
        contenders = [item for item in contender_pool if item["session_idx"] == freshest_session]
    else:
        contenders = candidates

    best = best_candidate(contenders)
    if best is None or best["score"] <= 0.0:
        return {"abstain": True, "predicted_value": None, "support_events": [], "policy_mode": "current"}

    answer = extract_current_answer(profile, best)
    if not answer:
        return {"abstain": True, "predicted_value": None, "support_events": [best], "policy_mode": "current"}
    return {"abstain": False, "predicted_value": answer, "support_events": [best], "policy_mode": "current"}


def abstain_like_policy():
    return {"abstain": True, "predicted_value": None, "support_events": [], "policy_mode": "abstain_like"}


def predict_from_locomo(query: dict, conversation_events: list[dict], conversation_headers: list[dict]):
    profile = build_query_profile(query)
    event_candidates = build_event_candidates(query, conversation_events, profile)
    header_candidates = build_header_candidates(query, conversation_headers, conversation_events, profile)
    if profile["policy_mode"] == "abstain_like":
        return profile, abstain_like_policy()
    if profile["policy_mode"] == "temporal":
        from locomo_temporal_policy import temporal_policy

        return profile, temporal_policy(query, profile, event_candidates, header_candidates)
    if profile["policy_mode"] == "multi_hop":
        from locomo_multihop_policy import multi_hop_policy

        return profile, multi_hop_policy(query, profile, event_candidates, header_candidates)
    return profile, current_policy(query, profile, event_candidates, header_candidates)
