import argparse
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SLOT_TYPES = [
    "location",
    "occupation",
    "education",
    "relationship",
    "plan",
    "preference",
    "possession",
    "health",
    "event",
    "date",
    "other",
]

CURRENT_KEYWORDS = (" now", " currently", " at present", " these days", " current ")
HISTORICAL_KEYWORDS = (" used to", " previously", " before that", " in the past", " former ")
TEMPORAL_KEYWORDS = (" before ", " after ", " when ", " last time", " first ", " later ", " earlier ")
OPEN_ANSWER_MARKERS = (
    "not mentioned",
    "no information",
    "unknown",
    "cannot be determined",
    "not enough information",
    "not provided",
)
QUESTION_WORDS = {
    "what",
    "when",
    "where",
    "who",
    "which",
    "why",
    "how",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt LoCoMo conversations into UCMD-friendly events, headers, and queries.")
    parser.add_argument("--input", default="../data/locomo10.json", help="Path to the LoCoMo JSON file.")
    parser.add_argument("--output-dir", default="locomo_adapted", help="Directory for JSONL adapter outputs.")
    parser.add_argument(
        "--locomo-version",
        default="locomo10_pinned_release_with_dec_2025_answer_fixes",
        help="Exact pinned LoCoMo release identifier to store in metadata.",
    )
    parser.add_argument("--adapter-version", default="locomo_adapter_v1", help="Adapter version string.")
    return parser.parse_args()


def maybe_git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_samples(path: Path):
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("data", "samples", "conversations"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return [payload]


def write_jsonl(path: Path, items):
    with path.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def slugify(text: str):
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "unknown"


def clean_text(text):
    return " ".join(str(text).replace("\t", " ").split())


def normalize_dia_ids(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, str):
        matches = re.findall(r"D\d+:\d+", value)
        if matches:
            return matches
        return [clean_text(value)] if clean_text(value) else []
    if isinstance(value, list):
        out = []
        seen = set()
        for item in value:
            for dia_id in normalize_dia_ids(item):
                if dia_id in seen:
                    continue
                seen.add(dia_id)
                out.append(dia_id)
        return out
    if isinstance(value, dict):
        for key in ("dia_ids", "dialogue_ids", "dialog_ids", "evidence", "source_dia_ids"):
            if key in value:
                return normalize_dia_ids(value[key])
    return []


def sample_id(sample, idx: int):
    for key in ("sample_id", "conversation_id", "id", "conv_id"):
        if sample.get(key):
            return str(sample[key])
    return f"locomo_{idx:04d}"


def speaker_map(sample: dict):
    speakers = sample.get("speakers")
    if isinstance(speakers, dict):
        return {slugify(key): clean_text(value) for key, value in speakers.items()}
    if isinstance(speakers, list):
        out = {}
        for idx, value in enumerate(speakers):
            if isinstance(value, dict):
                label = value.get("id") or value.get("speaker") or value.get("name") or f"speaker_{idx}"
                name = value.get("name") or value.get("speaker") or label
            else:
                label = f"speaker_{chr(ord('a') + idx)}"
                name = str(value)
            out[slugify(label)] = clean_text(name)
        if out:
            return out

    out = {}
    for key in ("speaker_a", "speaker_b"):
        if sample.get(key):
            out[key] = clean_text(sample[key])
    conversation = sample.get("conversation")
    if isinstance(conversation, dict):
        for key in ("speaker_a", "speaker_b"):
            if conversation.get(key):
                out[key] = clean_text(conversation[key])
    if out:
        return out
    return {"speaker_a": "speaker_a", "speaker_b": "speaker_b"}


def extract_sessions(sample: dict):
    if isinstance(sample.get("sessions"), list):
        return sample["sessions"]
    conversation = sample.get("conversation")
    if isinstance(conversation, dict) and isinstance(conversation.get("sessions"), list):
        return conversation["sessions"]
    if isinstance(conversation, dict):
        sessions = []
        session_numbers = sorted(
            {
                int(match.group(1))
                for key in conversation
                for match in [re.fullmatch(r"session_(\d+)", key)]
                if match
            }
        )
        for idx in session_numbers:
            sessions.append(
                {
                    "session_idx": idx,
                    "session_time": conversation.get(f"session_{idx}_date_time"),
                    "turns": conversation.get(f"session_{idx}", []),
                }
            )
        return sessions
    if isinstance(conversation, list):
        return conversation
    return []


def session_time(session: dict):
    for key in ("session_time", "time", "timestamp", "date"):
        if isinstance(session, dict) and session.get(key):
            return session[key]
    return None


def session_turns(session: dict):
    if not isinstance(session, dict):
        return []
    if isinstance(session.get("turns"), list):
        return session["turns"]
    for key in ("turns", "dialogue", "dialog", "conversation"):
        value = session.get(key)
        if isinstance(value, list):
            return value
    return []


def lookup_session_value(sample: dict, session_idx: int, base_key: str):
    direct_key = f"session_{session_idx}_{base_key}"
    if direct_key in sample:
        return sample[direct_key]

    container = sample.get(base_key)
    if isinstance(container, list):
        if session_idx - 1 < len(container):
            return container[session_idx - 1]
    if isinstance(container, dict):
        for key in (
            str(session_idx),
            f"session_{session_idx}",
            f"s{session_idx}",
            f"session{session_idx}",
            f"session_{session_idx}_{base_key}",
        ):
            if key in container:
                return container[key]
    return None


def normalize_text_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [{"text": clean_text(value), "dia_ids": []}]
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and not isinstance(value[1], dict)
    ):
        return [{"text": clean_text(value[0]), "dia_ids": normalize_dia_ids(value[1])}]
    if isinstance(value, dict):
        text = value.get("text") or value.get("summary") or value.get("observation") or value.get("content")
        if text:
            return [{"text": clean_text(text), "dia_ids": normalize_dia_ids(value)}]
        out = []
        for nested in value.values():
            out.extend(normalize_text_items(nested))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(normalize_text_items(item))
        return out
    return []


def normalize_summary_text(value):
    items = normalize_text_items(value)
    if not items:
        return None
    return items[0]["text"]


def normalize_qa_items(sample: dict):
    value = sample.get("qa")
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "questions", "qa"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
    return []


def resolve_entity(text: str, speakers: dict):
    lowered = f" {text.lower()} "
    for label, name in speakers.items():
        if not name:
            continue
        name_lower = name.lower()
        if f" {name_lower} " in lowered or f" {name_lower}'s " in lowered:
            return label, name

    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    for phrase in proper_nouns:
        phrase_lower = phrase.lower()
        if phrase_lower in QUESTION_WORDS:
            continue
        if phrase_lower not in {name.lower() for name in speakers.values()}:
            return slugify(phrase), phrase

    first_label = next(iter(speakers))
    return first_label, speakers[first_label]


def clean_value(value: str):
    value = value.strip(" .,:;")
    value = re.sub(r"^(an?|the)\s+", "", value, flags=re.IGNORECASE)
    return clean_text(value)


def infer_attribute_and_value(text: str):
    patterns = [
        ("occupation", [r"\bworks as (?:an? |the )?(?P<value>[^.]+)", r"\bjob is (?:an? |the )?(?P<value>[^.]+)"]),
        ("location", [r"\blives in (?P<value>[^.]+)", r"\bis in (?P<value>[^.]+)", r"\bmoved to (?P<value>[^.]+)"]),
        ("education", [r"\bstudies? (?P<value>[^.]+)", r"\battends? (?P<value>[^.]+ university[^.]*)"]),
        ("relationship", [r"\bmarried to (?P<value>[^.]+)", r"\bdating (?P<value>[^.]+)", r"\bfriend(?:s)? with (?P<value>[^.]+)"]),
        ("plan", [r"\bplans? to (?P<value>[^.]+)", r"\bgoing to (?P<value>[^.]+)"]),
        ("preference", [r"\blikes? (?P<value>[^.]+)", r"\bfavorite (?P<value>[^.]+)", r"\bprefers? (?P<value>[^.]+)"]),
        ("possession", [r"\bhas (?:a|an|the) (?P<value>[^.]+)", r"\bowns? (?P<value>[^.]+)"]),
        ("health", [r"\ballergic to (?P<value>[^.]+)", r"\bsuffers? from (?P<value>[^.]+)", r"\binjured (?P<value>[^.]+)"]),
        ("date", [r"\bon (?P<value>[A-Z][a-z]+ \d{1,2}(?:, \d{4})?)", r"\bdate is (?P<value>[^.]+)"]),
        (
            "event",
            [
                r"\bcelebrat(?:e|ed|ing) (?P<value>[^.]+)",
                r"\bwent to (?P<value>[^.]+)",
                r"\bvisited (?P<value>[^.]+)",
                r"\battended (?P<value>[^.]+)",
                r"\bjoined (?P<value>[^.]+)",
                r"\bvolunteered? at (?P<value>[^.]+)",
            ],
        ),
    ]
    for attribute, regexes in patterns:
        for regex in regexes:
            match = re.search(regex, text, flags=re.IGNORECASE)
            if match:
                value = clean_value(match.group("value"))
                if value:
                    return attribute, value

    def has_keyword(keyword: str):
        if " " in keyword:
            return keyword in lowered
        return re.search(rf"\b{re.escape(keyword)}\b", lowered) is not None

    keyword_fallbacks = [
        ("occupation", ("work", "job", "manager", "engineer", "designer", "teacher", "doctor")),
        ("location", ("live", "city", "town", "moved", "from", "home country", "based in")),
        ("education", ("school", "college", "university", "degree", "study")),
        ("relationship", ("friend", "boyfriend", "girlfriend", "married", "husband", "wife", "partner")),
        ("plan", ("plan", "trip", "travel", "visit", "going to")),
        ("preference", ("like", "favorite", "prefer", "enjoy")),
        ("possession", ("has ", "own", "bought", "car", "house", "dog", "cat")),
        ("health", ("allergy", "allergic", "doctor", "hospital", "pain", "injury")),
        ("event", ("party", "wedding", "meeting", "concert", "birthday")),
        ("date", ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january")),
    ]
    lowered = text.lower()
    for attribute, keywords in keyword_fallbacks:
        if any(has_keyword(keyword) for keyword in keywords):
            return attribute, clean_text(text)
    return "other", clean_text(text)


def extract_query_mode(question: str, category=None, speakers: dict | None = None):
    q = f" {question.lower()} "
    if any(keyword in q for keyword in CURRENT_KEYWORDS):
        return "current"
    if any(keyword in q for keyword in TEMPORAL_KEYWORDS):
        return "temporal"
    if any(keyword in q for keyword in HISTORICAL_KEYWORDS):
        return "historical"

    if any(phrase in q for phrase in (" not mention", " any mention", " any information", " do we know", " was there ever ")):
        return "adversarial"

    speaker_mentions = 0
    if speakers:
        speaker_mentions = sum(1 for name in speakers.values() if name and name.lower() in q)
    if speaker_mentions > 1 or any(phrase in q for phrase in (" both ", " together ", " each other ")):
        return "multi_hop"
    if category in {2}:
        return "temporal"
    if category in {5}:
        return "adversarial"
    if category in {3, 4}:
        return "multi_hop"
    if category in {1}:
        return "current"
    return "current"


def query_time_bucket(query_mode: str):
    if query_mode == "current":
        return "current"
    if query_mode == "historical":
        return "historical"
    return "historical"


def infer_query_entity(question: str, speakers: dict):
    label, _ = resolve_entity(question, speakers)
    return label


def infer_query_attribute(question: str):
    lowered = question.lower()
    if lowered.startswith("when "):
        return "date"
    if lowered.startswith("where "):
        return "location"
    if lowered.startswith("who "):
        return "relationship"
    attribute, _ = infer_attribute_and_value(question)
    return attribute


def adapt_locomo_conversation(sample: dict, sample_idx: int, adapter_version: str = "locomo_adapter_v1"):
    conversation_id = sample_id(sample, sample_idx)
    speakers = speaker_map(sample)
    sessions = extract_sessions(sample)
    events = []
    headers = []
    queries = []

    for idx, session in enumerate(sessions, start=1):
        obs_items = normalize_text_items(lookup_session_value(sample, idx, "observation"))
        session_event_ids = []
        for obs_idx, obs in enumerate(obs_items, start=1):
            text = obs["text"]
            if not text:
                continue
            entity, _ = resolve_entity(text, speakers)
            attribute, value = infer_attribute_and_value(text)
            event_id = f"locomo_{conversation_id}_s{idx}_o{obs_idx}"
            event = {
                "event_id": event_id,
                "conversation_id": conversation_id,
                "session_idx": idx,
                "session_time": session_time(session),
                "dia_ids": obs["dia_ids"],
                "speaker_owner": entity,
                "entity": entity,
                "attribute": attribute,
                "value": value,
                "regime": "global",
                "timestamp": idx,
                "time_bucket": "historical",
                "source": "locomo_observation",
                "text": text,
                "provenance": {
                    "kind": "locomo_observation",
                    "tool": "official_observation_release",
                    "actor": adapter_version,
                },
            }
            events.append(event)
            session_event_ids.append(event_id)

        summary_container = sample.get("session_summary")
        summary_text = normalize_summary_text(
            (summary_container or {}).get(f"session_{idx}_summary")
            if isinstance(summary_container, dict)
            else None
        )
        if summary_text:
            headers.append(
                {
                    "header_id": f"locomo_{conversation_id}_s{idx}",
                    "conversation_id": conversation_id,
                    "session_idx": idx,
                    "entity_scope": sorted(set(speakers.keys())),
                    "regime": "global",
                    "summary_text": summary_text,
                    "source": "locomo_session_summary",
                    "backlinks": session_event_ids,
                }
            )

    for q_idx, qa in enumerate(normalize_qa_items(sample), start=1):
        question = clean_text(qa.get("question") or qa.get("query") or qa.get("text") or "")
        answer = clean_text(qa.get("answer") or qa.get("gold_answer") or "")
        evidence_dia_ids = normalize_dia_ids(qa.get("evidence") or qa.get("evidence_dia_ids") or qa.get("supporting_dia_ids"))
        query_mode = extract_query_mode(question, qa.get("category"), speakers)
        queries.append(
            {
                "query_id": f"locomo_{conversation_id}_q{q_idx}",
                "conversation_id": conversation_id,
                "text": question,
                "entity": infer_query_entity(question, speakers),
                "attribute": infer_query_attribute(question),
                "regime": "global",
                "time_bucket": query_time_bucket(query_mode),
                "query_mode": query_mode,
                "category": qa.get("category"),
                "gold_answer": answer,
                "gold_evidence_dia_ids": evidence_dia_ids,
            }
        )

    return {"events": events, "headers": headers, "queries": queries}


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(input_path)
    all_events = []
    all_headers = []
    all_queries = []
    query_mode_counts = Counter()
    attribute_counts = Counter()

    for idx, sample in enumerate(samples, start=1):
        adapted = adapt_locomo_conversation(sample, idx, adapter_version=args.adapter_version)
        all_events.extend(adapted["events"])
        all_headers.extend(adapted["headers"])
        all_queries.extend(adapted["queries"])
        query_mode_counts.update(query["query_mode"] for query in adapted["queries"])
        attribute_counts.update(event["attribute"] for event in adapted["events"])

    write_jsonl(output_dir / "events.jsonl", all_events)
    write_jsonl(output_dir / "headers.jsonl", all_headers)
    write_jsonl(output_dir / "queries.jsonl", all_queries)
    metadata = {
        "adapter_version": args.adapter_version,
        "locomo_version": args.locomo_version,
        "source_path": str(input_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": maybe_git_commit(),
        "conversation_count": len(samples),
        "event_count": len(all_events),
        "header_count": len(all_headers),
        "query_count": len(all_queries),
        "slot_taxonomy": SLOT_TYPES,
        "query_mode_counts": dict(sorted(query_mode_counts.items())),
        "attribute_counts": dict(sorted(attribute_counts.items())),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")

    print(f"wrote {len(all_events)} events to {output_dir / 'events.jsonl'}")
    print(f"wrote {len(all_headers)} headers to {output_dir / 'headers.jsonl'}")
    print(f"wrote {len(all_queries)} queries to {output_dir / 'queries.jsonl'}")
    print(f"wrote metadata to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
