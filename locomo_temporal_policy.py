import re
from datetime import datetime, timedelta


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_session_datetime(text: str | None):
    if not text:
        return None
    cleaned = text.replace(" am ", " AM ").replace(" pm ", " PM ")
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def date_phrase(dt: datetime):
    return f"{dt.day} {dt.strftime('%B %Y')}"


def month_phrase(dt: datetime):
    return f"{dt.strftime('%B %Y')}"


def previous_month(dt: datetime):
    year = dt.year if dt.month > 1 else dt.year - 1
    month = dt.month - 1 if dt.month > 1 else 12
    return datetime(year, month, 1)


def next_month(dt: datetime):
    year = dt.year if dt.month < 12 else dt.year + 1
    month = dt.month + 1 if dt.month < 12 else 1
    return datetime(year, month, 1)


def previous_weekday(anchor: datetime, weekday: int):
    days_back = (anchor.weekday() - weekday) % 7
    days_back = 7 if days_back == 0 else days_back
    return anchor - timedelta(days=days_back)


def temporal_answer_from_text(text: str, session_time: str | None):
    text = text or ""
    match = re.search(r"\b(\d{1,2}\s+[A-Z][a-z]+(?:,\s+\d{4})?)\b", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([A-Z][a-z]+\s+\d{4})\b", text)
    if match:
        return match.group(1)

    dt = parse_session_datetime(session_time)
    lowered = text.lower()
    if "last year" in lowered and dt is not None:
        return str(dt.year - 1)
    if "this year" in lowered and dt is not None:
        return str(dt.year)
    if "next month" in lowered and dt is not None:
        return month_phrase(next_month(dt))
    if "this month" in lowered and dt is not None:
        return month_phrase(dt)
    if "last month" in lowered and dt is not None:
        return month_phrase(previous_month(dt))
    if ("last week" in lowered or "the week before" in lowered) and dt is not None:
        return f"The week before {date_phrase(dt)}"
    if ("last weekend" in lowered or "over the weekend" in lowered) and dt is not None:
        return f"The weekend before {date_phrase(dt)}"
    if "two weekends ago" in lowered and dt is not None:
        return f"two weekends before {date_phrase(dt)}"
    if "yesterday" in lowered and dt is not None:
        return date_phrase(dt - timedelta(days=1))
    for weekday_name, weekday_idx in WEEKDAYS.items():
        if f"last {weekday_name}" in lowered and dt is not None:
            return date_phrase(previous_weekday(dt, weekday_idx))
    if "recently" in lowered and dt is not None and re.search(r"\bwhen\b|\bdate\b|\btime\b", lowered):
        return date_phrase(dt)
    return None


def temporal_policy(query: dict, profile: dict, event_candidates, header_candidates):
    _ = query
    _ = profile
    candidates = sorted(event_candidates + header_candidates, key=lambda item: (item["score"], item["session_idx"]), reverse=True)
    for candidate in candidates[:8]:
        answer = temporal_answer_from_text(candidate["text"], candidate.get("session_time"))
        if answer:
            return {
                "abstain": False,
                "predicted_value": answer,
                "support_events": [candidate],
                "policy_mode": "temporal",
            }
    return {"abstain": True, "predicted_value": None, "support_events": candidates[:1], "policy_mode": "temporal"}
