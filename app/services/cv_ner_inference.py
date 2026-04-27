"""
CV NER inference — runs the fine-tuned XLM-RoBERTa model on CV text and
returns extracted SKILL and TITLE spans.

The model is lazy-loaded as a module-level singleton on first call; subsequent
calls reuse the in-memory pipeline.

Long texts are processed in overlapping windows because XLM-R caps at 512
subword tokens. Predictions across windows are deduplicated by surface form.

Usage:
    from app.services.cv_ner_inference import extract_entities
    result = extract_entities(cv_text)
    # → {"skills": [{"text": "Python", "score": 0.95, "start": 12, "end": 18}, ...],
    #    "titles": [{"text": "Software Engineer", "score": 0.91, ...}]}
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "models" / "cv_ner" / "cv_ner_final"
LEXICON_PATH = ROOT / "data" / "lexicons" / "esco_lexicon.json"

# Option-B quality thresholds for filtering NER skill predictions
_HIGH_CONFIDENCE = 0.70   # always keep
_LOW_CONFIDENCE = 0.55    # keep only if present in ESCO lexicon

# Window settings — XLM-R max length is 512 subword tokens. We use a smaller
# window with overlap so entities near a boundary aren't truncated.
_WINDOW_CHARS = 1800       # ~400 tokens worth of typical CV text
_OVERLAP_CHARS = 200       # ~40 tokens overlap

# Confidence floor: drop spans below this score
_MIN_SCORE = 0.55

_pipeline = None
_pipeline_lock = threading.Lock()
_load_failed = False


def _get_pipeline():
    """Lazy-load the HF token-classification pipeline."""
    global _pipeline, _load_failed
    if _pipeline is not None:
        return _pipeline
    if _load_failed:
        return None
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        if _load_failed:
            return None
        try:
            from transformers import pipeline
            if not MODEL_DIR.exists():
                logger.warning("cv_ner model not found at %s", MODEL_DIR)
                _load_failed = True
                return None
            _pipeline = pipeline(
                task="token-classification",
                model=str(MODEL_DIR),
                tokenizer=str(MODEL_DIR),
                aggregation_strategy="simple",
                device=-1,  # CPU; pipeline auto-uses GPU if available via accelerate
            )
            logger.info("cv_ner model loaded from %s", MODEL_DIR)
        except Exception as exc:
            logger.exception("cv_ner model load failed: %s", exc)
            _load_failed = True
            return None
    return _pipeline


def _chunk_text(text: str) -> list[tuple[str, int]]:
    """Split text into overlapping windows. Returns list of (chunk, char_offset)."""
    text = text or ""
    if len(text) <= _WINDOW_CHARS:
        return [(text, 0)]
    windows: list[tuple[str, int]] = []
    step = _WINDOW_CHARS - _OVERLAP_CHARS
    pos = 0
    while pos < len(text):
        windows.append((text[pos : pos + _WINDOW_CHARS], pos))
        if pos + _WINDOW_CHARS >= len(text):
            break
        pos += step
    return windows


def _normalize_span_text(s: str) -> str:
    return " ".join(s.split()).strip()


def _run_pipeline_on_text(text: str) -> list[dict[str, Any]]:
    """Run the NER pipeline over the full text (chunked) and return raw spans
    with absolute character offsets."""
    pipe = _get_pipeline()
    if pipe is None or not text:
        return []

    raw_results: list[dict[str, Any]] = []
    for chunk, offset in _chunk_text(text):
        try:
            preds = pipe(chunk)
        except Exception as exc:
            logger.warning("cv_ner pipeline failed on chunk: %s", exc)
            continue
        for p in preds:
            entity_group = p.get("entity_group") or p.get("entity") or ""
            if not entity_group:
                continue
            score = float(p.get("score") or 0.0)
            if score < _MIN_SCORE:
                continue
            text_span = _normalize_span_text(str(p.get("word") or ""))
            if not text_span:
                continue
            raw_results.append({
                "entity": entity_group,         # "SKILL" or "TITLE"
                "text": text_span,
                "score": score,
                "start": int(p.get("start", 0)) + offset,
                "end": int(p.get("end", 0)) + offset,
            })
    return raw_results


def _dedupe_by_text(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-scoring instance per (entity, lowercased text)."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for s in spans:
        key = (s["entity"], s["text"].lower())
        if key not in best or s["score"] > best[key]["score"]:
            best[key] = s
    return sorted(best.values(), key=lambda x: -x["score"])


def extract_entities(text: str) -> dict[str, list[dict[str, Any]]]:
    """
    Run CV NER on the given text and return grouped predictions.

    Returns:
        {
            "skills": [{"text": ..., "score": ..., "start": ..., "end": ...}, ...],
            "titles": [...],
        }

    Both lists are sorted by descending confidence and deduplicated by
    case-folded surface form.
    """
    raw = _run_pipeline_on_text(text or "")
    if not raw:
        return {"skills": [], "titles": []}

    deduped = _dedupe_by_text(raw)
    skills = [
        {"text": s["text"], "score": s["score"], "start": s["start"], "end": s["end"]}
        for s in deduped if s["entity"] == "SKILL"
    ]
    titles = [
        {"text": s["text"], "score": s["score"], "start": s["start"], "end": s["end"]}
        for s in deduped if s["entity"] == "TITLE"
    ]
    return {"skills": skills, "titles": titles}


def is_available() -> bool:
    """Check if the model is loadable without actually loading it (cheap)."""
    return MODEL_DIR.exists() and not _load_failed


# ---- ESCO lexicon for low-confidence filtering ---------------------------

_ESCO_LEXICON_KEYS: set[str] | None = None
_ESCO_LEXICON_LOCK = threading.Lock()
_ESCO_LOAD_FAILED = False


def _load_esco_lexicon_keys() -> set[str]:
    """Lazy-load the ESCO lexicon, returning just the set of normalized keys
    (we only need membership testing, not the full entry data)."""
    global _ESCO_LEXICON_KEYS, _ESCO_LOAD_FAILED
    if _ESCO_LEXICON_KEYS is not None:
        return _ESCO_LEXICON_KEYS
    if _ESCO_LOAD_FAILED:
        return set()
    with _ESCO_LEXICON_LOCK:
        if _ESCO_LEXICON_KEYS is not None:
            return _ESCO_LEXICON_KEYS
        if _ESCO_LOAD_FAILED:
            return set()
        try:
            if not LEXICON_PATH.exists():
                logger.warning("ESCO lexicon not found at %s", LEXICON_PATH)
                _ESCO_LOAD_FAILED = True
                return set()
            with LEXICON_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", {})
            _ESCO_LEXICON_KEYS = set(entries.keys())
            logger.info("ESCO lexicon loaded: %d entries", len(_ESCO_LEXICON_KEYS))
        except Exception as exc:
            logger.exception("ESCO lexicon load failed: %s", exc)
            _ESCO_LOAD_FAILED = True
            return set()
    return _ESCO_LEXICON_KEYS


def _normalize_for_lexicon(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def extract_skills_filtered(text: str) -> list[dict[str, Any]]:
    """
    Run NER and apply the Option-B quality filter:
      - score >= 0.70 → always keep
      - 0.55 <= score < 0.70 → keep only if present in ESCO lexicon
      - score < 0.55 → drop (already filtered upstream)

    Returns a list of skill_row dicts compatible with cv_parser._merge_skill_rows:
        {"skill", "confidence", "source", "evidence"}
    """
    result = extract_entities(text or "")
    raw_skills = result.get("skills", [])
    if not raw_skills:
        return []

    lex_keys = _load_esco_lexicon_keys()  # may be empty if load failed; that's OK

    rows: list[dict[str, Any]] = []
    for span in raw_skills:
        skill_text = str(span.get("text") or "").strip()
        if not skill_text:
            continue
        try:
            score = float(span.get("score") or 0.0)
        except Exception:
            continue

        if score >= _HIGH_CONFIDENCE:
            keep = True
        elif score >= _LOW_CONFIDENCE:
            keep = _normalize_for_lexicon(skill_text) in lex_keys
        else:
            keep = False
        if not keep:
            continue

        rows.append({
            "skill": skill_text,
            "confidence": score,
            # Use "ner_span:" prefix so _source_channel_family() classifies
            # this as the "ner" channel (matches existing infrastructure).
            "source": "ner_span:cv_ner_finetuned",
            "evidence": [skill_text],
        })
    return rows
