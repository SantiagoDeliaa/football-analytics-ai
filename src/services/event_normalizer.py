from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _find_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    if match.groups():
        return match.group(1).strip()
    return match.group(0).strip()


def _extract_formations(text: str) -> list[str]:
    formations = re.findall(r"\b(?:[3-5]-[1-5]-[1-5](?:-[1-3])?)\b", text)
    unique: list[str] = []
    for formation in formations:
        if formation not in unique:
            unique.append(formation)
    return unique[:8]


def _keyword_summary(text: str, keywords: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    counts = {keyword: lowered.count(keyword.lower()) for keyword in keywords}
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "signals": counts,
        "top_signal": top[0][0] if top and top[0][1] > 0 else "not_detected",
    }


def normalize_event_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(raw_data.get("raw_text", "") or "")
    page_count = int(raw_data.get("page_count", 0) or 0)
    status = raw_data.get("status", "error")
    file_name = raw_data.get("file_name", "unknown.pdf")

    teams_line = _find_first(r"([A-Za-zÀ-ÿ0-9 ._-]+\s+vs\.?\s+[A-Za-zÀ-ÿ0-9 ._-]+)", raw_text)
    score_line = _find_first(r"\b(\d{1,2}\s*[-:]\s*\d{1,2})\b", raw_text)
    competition = _find_first(r"(?:competition|league|torneo)\s*[:\-]\s*([^\n]+)", raw_text) or "unknown"
    report_date = _find_first(r"(?:date|fecha)\s*[:\-]\s*([^\n]+)", raw_text) or "unknown"
    formations = _extract_formations(raw_text)

    section_hits = Counter(
        {
            "attack": len(re.findall(r"\battack|offen|chance|xg|shot|final third\b", raw_text, flags=re.IGNORECASE)),
            "defense": len(re.findall(r"\bdefen|press|duel|recover|block\b", raw_text, flags=re.IGNORECASE)),
            "transitions": len(re.findall(r"\btransition|counter|turnover|regain\b", raw_text, flags=re.IGNORECASE)),
            "build_up": len(re.findall(r"\bbuild ?up|progression|possession|pass network\b", raw_text, flags=re.IGNORECASE)),
            "finishing": len(re.findall(r"\bfinish|conversion|on target|goal\b", raw_text, flags=re.IGNORECASE)),
        }
    )

    normalized: dict[str, Any] = {
        "status": "ok" if status == "ok" else "warning",
        "match_info": {
            "file_name": file_name,
            "competition": competition,
            "date": report_date,
            "teams": teams_line or "unknown",
            "score": score_line or "unknown",
            "source_pages": page_count,
        },
        "team_summary": {
            "detected_teams_text": teams_line or "unknown",
            "total_characters": len(raw_text),
            "estimated_sections": max(1, raw_text.count("\n\n")),
        },
        "formations": formations if formations else ["not_detected"],
        "attack": _keyword_summary(raw_text, ["xg", "shots", "box entries", "crosses", "final third"]),
        "defense": _keyword_summary(raw_text, ["pressing", "duels", "recoveries", "blocks", "interceptions"]),
        "transitions": _keyword_summary(raw_text, ["counter", "turnover", "regain", "direct attack", "transition"]),
        "build_up": _keyword_summary(raw_text, ["build up", "progression", "pass network", "possession", "switch"]),
        "finishing": _keyword_summary(raw_text, ["conversion", "on target", "goal", "big chances", "xg"]),
        "meta": {
            "raw_status": status,
            "sections_detected": dict(section_hits),
            "fallback_applied": status != "ok",
        },
    }
    if status != "ok":
        normalized["status"] = "warning"
    return normalized
