"""Heuristic memory extraction and lexical retrieval for the offline MVP."""

from __future__ import annotations

import re
from typing import Any

from .db import Database, utc_now


STOP_WORDS = {"the", "and", "that", "this", "with", "from", "about", "what", "have", "你", "我", "的", "是"}


def _terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(term) > 1 and term.lower() not in STOP_WORDS}


def extract_memory_candidates(text: str) -> list[dict[str, str]]:
    """Extract only explicit, high-signal statements; one-off chat is ignored."""
    patterns = [
        (r"\b(?:remember that|remember)\s+(.+)", "fact"),
        (r"\bI\s+(?:am|study|work as|live in|have)\s+(.+)", "fact"),
        (r"\bI\s+(?:like|love|enjoy|prefer)\s+(.+)", "preference"),
        (r"\b我的(?:偏好是|专业是|项目是)\s*(.+)", "preference"),
        (r"\b我(?:喜欢|偏好|正在学习|住在)\s*(.+)", "preference"),
    ]
    candidates: list[dict[str, str]] = []
    for pattern, memory_type in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip().rstrip("。.!！")
            if content:
                candidates.append({"type": memory_type, "content": content})
    return candidates


def save_extracted(db: Database, text: str) -> list[dict[str, Any]]:
    saved = []
    now = utc_now()
    with db.connection() as conn:
        for candidate in extract_memory_candidates(text):
            existing = conn.execute(
                "SELECT * FROM memories WHERE lower(content) = lower(?)", (candidate["content"],)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE memories SET evidence_count=evidence_count+1, confidence=min(1, confidence+0.05), "
                    "stability=min(1, stability+0.03), last_updated=? WHERE id=?",
                    (now, existing["id"]),
                )
                # Re-read the row so the API reflects incremented evidence.
                updated = conn.execute("SELECT * FROM memories WHERE id=?", (existing["id"],)).fetchone()
                saved.append(dict(updated))
                continue
            memory_id = db.new_id()
            conn.execute(
                "INSERT INTO memories(id,type,content,confidence,stability,source,evidence_count,created_at,last_updated) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (memory_id, candidate["type"], candidate["content"], 0.65, 0.3, "explicit_heuristic", 1, now, now),
            )
            saved.append({"id": memory_id, **candidate, "confidence": 0.65, "stability": 0.3, "source": "explicit_heuristic", "evidence_count": 1, "created_at": now, "last_updated": now, "valid_until": None})
    return saved


def retrieve(db: Database, query: str, limit: int = 5) -> list[dict[str, Any]]:
    query_terms = _terms(query)
    with db.connection() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM memories ORDER BY last_updated DESC").fetchall()]
    scored = []
    for row in rows:
        overlap = len(query_terms & _terms(row["content"]))
        if overlap:
            scored.append((overlap, row["confidence"], row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # A lightweight semantic fallback for common communication questions. This
    # keeps explicit style preferences useful without requiring embeddings.
    if not scored and query_terms & {"answer", "answers", "response", "respond", "回答", "回复"}:
        scored = [(0, row["confidence"], row) for row in rows if row["type"] == "preference"]
        scored.sort(key=lambda item: item[1], reverse=True)
    return [item[2] for item in scored[:limit]]
