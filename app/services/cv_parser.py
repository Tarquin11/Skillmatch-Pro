import io
import re
import math
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable
import numpy as np
import pdfplumber
from docx import Document
from app.ai.skill_canonicalization import canonicalize_skill
from app.services.embedding_service import EmbeddingService

_SKILL_HEADING_HINTS = (
    "skills", "technical skills", "technologies", "tech stack",
    "tools", "competencies", "core skills"
)
_DURATION_PHRASE_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:months?|years?|yrs?)$", re.I)
WORD_RE = re.compile(r"[a-z0-9+.#/\-]+")
_EXPERIENCE_YEARS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|ans)\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(present|current|now|19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

_EXPERIENCE_HEADING_HINTS = (
    "experience",
    "work experience",
    "professional experience",
    "employment",
    "working",
    "internship",
    "internships",
    "projects",
    "project",
    "freelance",
)
_EDUCATION_HEADING_HINTS = (
    "education",
    "academic",
    "school",
    "university",
    "college",
)
_TITLE_KEYWORDS = (
    "engineer",
    "developer",
    "analyst",
    "manager",
    "designer",
    "scientist",
    "consultant",
    "specialist",
    "student",
    "intern",
    "architect",
)
BULLET_PREFIXES = ("-", "*", "•", "â€¢")

MAX_TEXT_CHARS = 120_000
MAX_LINES = 4_000
MAX_LINE_CHARS = 600
MAX_TOKENS = 8_000
MAX_NGRAM_TOKENS = 3_000
MAX_SECTION_PHRASES = 1_200
MAX_PHRASE_CHARS = 180
MAX_KNOWN_SKILLS = 5_000
DEFAULT_SKILL_TIME_BUDGET_SECONDS = 0.35


def _is_skill_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _SKILL_HEADING_HINTS)

def _is_noise_skill_phrase(phrase: str) -> bool:
    p = _normalize_text(phrase)
    return bool(_DURATION_PHRASE_RE.fullmatch(p))

def _clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text(file_bytes, filename):
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
            return _clean_extracted_text(text)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        return _clean_extracted_text(text)
    return ""
    
def _normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\t", " ")
    lines = [
        re.sub(r"[ ]{2,}", " ", line[:MAX_LINE_CHARS]).strip()
        for line in normalized.split("\n")[:MAX_LINES]
    ]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()[:MAX_TEXT_CHARS]

def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)

def _ngram_candidates(tokens: list[str], max_n: int = 4) -> set[str]:
    out = set(tokens)
    for n in range(2, max_n + 1):
        for i in range(0,max(len(tokens) -n + 1 , 0)):
            out.add(" ".join(tokens[i : i + n]))
            if len(out) >= MAX_TOKENS:
                return out
    return out

def _skill_key(value: str) -> str:
    canonical = canonicalize_skill(value)
    canonical = canonical.replace("c++", "cpp")
    canonical = canonical.replace("c#", "csharp")
    canonical = canonical.replace("+", " plus ")
    canonical = canonical.replace("#", " sharp ")
    return re.sub(r"[^a-z0-9]", "", canonical)

def _acronym(value: str) -> str:
    parts = [p for p in canonicalize_skill(value).split() if p]
    return "".join(p[0] for p in parts if p and p[0].isalnum())


def _looks_like_letter_spaced_text(value: str) -> bool:
    parts = [p for p in value.strip().split() if p]
    alpha_parts = [p for p in parts if any(ch.isalpha() for ch in p)]
    if len(alpha_parts) < 6:
        return False
    single = sum(1 for p in alpha_parts if len(p) == 1 and p.isalpha())
    return (single / len(alpha_parts)) >= 0.75


def _collapse_letter_spaced_text(value: str) -> str:
    parts = [p for p in value.strip().split() if p]
    if not parts:
        return ""
    if _looks_like_letter_spaced_text(value):
        return "".join(parts).upper()
    return value.strip()

def _build_skill_index(known_skills: Iterable[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for raw in known_skills or []:
        canonical = canonicalize_skill(raw)
        if not canonical:
            continue
        skill_key= _skill_key(canonical)
        if not skill_key:
            continue
        aliases = {skill_key}
        ac = _acronym(canonical)
        if len(ac) >= 3 and len(canonical.split()) >= 3:
            aliases.add(ac)
        
        index[canonical] = {"skill_key": skill_key, "aliases": aliases}
    return index

def _heading_candidate(raw_line: str) -> tuple[bool, str, float]:
    line = raw_line.strip()
    if not line:
        return False, "", 0.0
    for prefix in BULLET_PREFIXES:
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
            break

    trimmed = line.strip(":").strip()
    if not trimmed or len(trimmed) > 80:
        return False, "", 0.0

    words = trimmed.split()
    if len(words) > 6:
        return False, "", 0.0

    letters = [c for c in trimmed if c.isalpha()]
    if not letters:
        return False, "", 0.0

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    title_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    has_colon = raw_line.rstrip().endswith(":")

    is_heading = has_colon or upper_ratio >= 0.6 or title_ratio >= 0.8
    if not is_heading:
        return False, "", 0.0

    weight = 1.1 if has_colon or upper_ratio >= 0.75 else 1.0
    key = _normalize_text(trimmed)
    return True, key, weight


def _extract_sections(text: str) -> tuple[dict[str, list[str]], dict[str, float]]:
    sections: dict[str, list[str]] = {"other": []}
    weights: dict[str, float] = {"other": 0.85}
    current = "other"
    for raw_line in (text or "").splitlines()[:MAX_LINES]:
        line = raw_line[:MAX_LINE_CHARS].strip()
        if not line:
            continue
        is_heading, key, weight = _heading_candidate(line)
        if is_heading and key:
            current = key
            sections.setdefault(current, [])
            weights[current] = max(weights.get(current, 0.85), weight)
            continue
        sections.setdefault(current, []).append(line)
    return sections, weights

def _extract_section_phrases(sections: dict[str, list[str]]) -> list[tuple[str, str]]:
    phrases: list[tuple[str, str]] = []
    for section, lines in sections.items():
        for line in lines:
            line = line[:MAX_LINE_CHARS].strip()
            if not line:
                continue
            for prefix in BULLET_PREFIXES:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            parts = re.split(r"[,\|;/]+", line)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(part) > MAX_PHRASE_CHARS:
                    continue
                phrases.append((part, section))
                if len(phrases) >= MAX_SECTION_PHRASES:
                    return phrases
    return phrases


def _is_experience_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _EXPERIENCE_HEADING_HINTS)


def _is_education_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _EDUCATION_HEADING_HINTS)


def _year_from_token(token: str) -> int | None:
    token = (token or "").strip().lower()
    if token in {"present", "current", "now"}:
        return datetime.now(timezone.utc).year
    if re.fullmatch(r"(19\d{2}|20\d{2})", token):
        return int(token)
    return None

_EMBEDDER: EmbeddingService | None = None
_EMBED_CACHE: dict[int, tuple[np.ndarray, list[str]]] = {}

def _get_embedder() -> EmbeddingService | None:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        _EMBEDDER = EmbeddingService()
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER

def _get_skill_embeddings(known_skills: list[str]) -> tuple[np.ndarray | None, list[str]]:
    if not known_skills:
        return None, []
    cache_key = hash(tuple(sorted(known_skills)))
    cached = _EMBED_CACHE.get(cache_key)
    if cached:
        return cached
    embedder = _get_embedder()
    if not embedder:
        return None, []
    vectors = embedder.generate_embeddings(known_skills)
    arr = np.asarray(vectors, dtype=np.float32)
    _EMBED_CACHE[cache_key] = (arr, list(known_skills))
    return _EMBED_CACHE[cache_key]

def _calibrate_confidence(raw_conf : float, source: str, section_weight: float) -> float:
    if raw_conf <= 0:
        return 0.0

    raw_conf = max(0.001, min(0.999, raw_conf))
    source_key = (source or "").split(":")[0]
    source_weight = {
        "exact": 1.0,
        "synonym": 0.95,
        "fuzzy": 0.90,
        "semantic":0.92,
        "legacy": 0.88,
    }.get(source_key, 0.92)

    sec = max(0.75, min(1.05, section_weight))
    adjusted = raw_conf * source_weight * sec
    adjusted = max(0.001, min(0.999, adjusted))
    temp = 1.15 if source_key in ("fuzzy","semantic") else 1.0
    logit = math.log(adjusted / (1 - adjusted))
    calibrated = 1.0 / (1.0 + math.exp(-logit / temp))
    return max(0.01, min(0.99, calibrated))

def detect_skills_with_confidence(
    text: str,
    known_skills: Iterable[str],
    min_confidence: float = 0.6,
    use_semantic: bool = True,
    semantic_threshold: float = 0.65,
    time_budget_seconds: float | None = DEFAULT_SKILL_TIME_BUDGET_SECONDS,
) -> list[dict[str, Any]]:
    known_list = [
        str(k).strip()
        for k in known_skills
        if isinstance(k, str) and str(k).strip()
    ]
    if len(known_list) > MAX_KNOWN_SKILLS:
        known_list = known_list[:MAX_KNOWN_SKILLS]
    index = _build_skill_index(known_list)
    if not index:
        return []
    normalized_text = _normalize_text(text or "")
    tokens = _tokenize(normalized_text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    if not tokens:
        return []
    ngrams = _ngram_candidates(tokens[:MAX_NGRAM_TOKENS], max_n=4)
    ngram_keys = {g: _skill_key(g) for g in ngrams}
    ngram_acronyms = {g: _acronym(g) for g in ngrams}
    sections, section_weights = _extract_sections(text)
    section_phrases = _extract_section_phrases(sections)
    skill_vectors, skill_names = _get_skill_embeddings(known_list) if use_semantic else (None, [])
    deadline = None
    if time_budget_seconds and time_budget_seconds > 0:
        deadline = time.perf_counter() + float(time_budget_seconds)

    def _time_exceeded() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    semantic_hits: dict[str, tuple[float, str, float]] = {}
    if use_semantic and skill_vectors is not None and skill_names and not _time_exceeded():
        embedder = _get_embedder()
        if embedder:
            phrase_items: list[tuple[str, str]] = []
            seen_phrases: set[str] = set()
            for phrase, section in section_phrases:
                if _time_exceeded():
                    break
                key = _normalize_text(phrase)
                if not key or key in seen_phrases:
                    continue
                seen_phrases.add(key)
                phrase_items.append((phrase, section))
                if len(phrase_items) >= 200:
                    break

            if phrase_items:
                phrases = [p for p, _ in phrase_items]
                try:
                    phrase_vectors = np.asarray(
                        embedder.generate_embeddings(phrases),
                        dtype=np.float32,
                    )
                except Exception:
                    phrase_vectors = np.asarray([], dtype=np.float32)
                if phrase_vectors.size:
                    sims = phrase_vectors @ skill_vectors.T
                    for i, (_, section) in enumerate(phrase_items):
                        if _time_exceeded():
                            break
                        idx = int(np.argmax(sims[i]))
                        sim = float(sims[i, idx])
                        if sim < semantic_threshold:
                            continue
                        skill = skill_names[idx]
                        weight = section_weights.get(section, 0.85)
                        conf = min(0.92, sim * weight)
                        prev = semantic_hits.get(skill)
                        if prev is None or conf > prev[0]:
                            semantic_hits[skill] = (conf, f"semantic:{section}", weight)

    hits: list[dict[str, Any]] = []
    for canonical, meta in index.items():
        if _time_exceeded():
            break
        skill_key = meta["skill_key"]
        aliases = meta["aliases"]
        best_conf = 0.0
        best_source = "fuzzy"
        best_section_weight = 0.9
        is_short_ambiguous = len(skill_key) <= 1

        for gram in ngrams:
            if _time_exceeded():
                break
            if is_short_ambiguous:
                continue
            gram_key = ngram_keys[gram]
            if not gram_key:
                continue
            if gram_key in aliases:
                conf = 0.98 if gram_key == skill_key else 0.90
                source = "exact" if gram_key == skill_key else "synonym"
            elif (
                len(ngram_acronyms[gram]) >= 3
                and len(gram.split()) >= 3
                and ngram_acronyms[gram] in aliases
            ):
                conf = 0.88
                source = "synonym"
            else:
                ratio = SequenceMatcher(None, skill_key, gram_key).ratio()
                if ratio < 0.90:
                    continue
                conf = min(0.82, ratio)
                source = "fuzzy"
            if conf > best_conf:
                best_conf = conf
                best_source = source
                best_section_weight = 0.9

        for phrase, section in section_phrases:
            if _time_exceeded():
                break
            if _is_noise_skill_phrase(phrase):
                continue
            phrase_key = _skill_key(phrase)
            if not phrase_key:
                continue
            is_skill_section = _is_skill_heading(section)
            if is_short_ambiguous and not is_skill_section:
                continue
            if phrase_key in aliases:
                base = 0.95
                conf = min(0.99, base * section_weights.get(section, 0.85))
                source = f"exact:{section}"
            else:
                ratio = SequenceMatcher(None, skill_key, phrase_key).ratio()
                min_ratio = 0.90 if is_skill_section else 0.97
                if ratio < min_ratio:
                    continue
                if (not is_skill_section) and (len(skill_key) <= 3):
                    continue
                base = min(0.80, ratio)
                conf = min(0.90, base * section_weights.get(section, 0.85))
                source = f"fuzzy:{section}"
            if conf > best_conf:
                best_conf = conf
                best_source = source
                best_section_weight = section_weights.get(section, 0.85)

        semantic = semantic_hits.get(canonical)
        if semantic and semantic[0] > best_conf:
            best_conf, best_source, best_section_weight = semantic 

        if best_conf >= min_confidence:
            hits.append(
                {
                    "skill": canonical,
                    "confidence": round(best_conf, 2),
                    "source": best_source,
                }
            )
    hits.sort(key=lambda row: (-float(row["confidence"]), str(row["skill"])))
    return hits


def detect_title(text: str) -> str | None:
    lines = [
        re.sub(r"\s+", " ", ln[:MAX_LINE_CHARS]).strip()
        for ln in (text or "").splitlines()[:MAX_LINES]
    ]
    lines = [ln for ln in lines if ln]

    for line in lines[:40]:
        low = line.lower()
        if "@" in low or "http://" in low or "https://" in low:
            continue
        if len(line) > 140:
            continue
        if " with " in low and any(k in low for k in _TITLE_KEYWORDS):
            prefix = line.split(" with ", 1)[0].strip(" -,:")
            if 2 <= len(prefix.split()) <= 8:
                return prefix

    for line in lines[:25]:
        low = line.lower()
        if "@" in low:
            continue
        if len(line) > 90:
            continue
        if any(k in low for k in _TITLE_KEYWORDS):
            return line.strip(" -,:")
        if _looks_like_letter_spaced_text(line):
            collapsed = _collapse_letter_spaced_text(line)
            if collapsed:
                return collapsed
    return None


def detect_experience_years(text: str) -> float | None:
    normalized = _normalize_text(text or "")
    explicit = [float(x) for x in _EXPERIENCE_YEARS_RE.findall(normalized)]
    if explicit:
        return round(max(explicit), 2)

    sections, _weights = _extract_sections(text)
    ranges: list[tuple[int, int]] = []
    current_year = datetime.now(timezone.utc).year

    for section, lines in sections.items():
        if _is_education_heading(section):
            continue
        heading_bonus = _is_experience_heading(section)
        for line in lines:
            line_low = _normalize_text(line)
            if not heading_bonus and not any(h in line_low for h in _EXPERIENCE_HEADING_HINTS):
                continue
            for start_s, end_s in _YEAR_RANGE_RE.findall(line_low):
                start = _year_from_token(start_s)
                end = _year_from_token(end_s)
                if start is None or end is None:
                    continue
                end = min(end, current_year)
                if end < start:
                    continue
                if (end - start) > 50:
                    continue
                ranges.append((start, end))

    if not ranges:
        lines = [
            _normalize_text(ln[:MAX_LINE_CHARS])
            for ln in (text or "").splitlines()[:MAX_LINES]
            if ln and ln.strip()
        ]
        context_years: list[int] = []
        for i, line in enumerate(lines):
            year_tokens = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", line)]
            if not year_tokens:
                continue
            window_start = max(0, i - 12)
            window_end = min(len(lines), i + 13)
            window = " ".join(lines[window_start:window_end])
            line_has_experience = any(h in line for h in _EXPERIENCE_HEADING_HINTS)
            line_has_education = any(h in line for h in _EDUCATION_HEADING_HINTS)
            window_has_experience = any(h in window for h in _EXPERIENCE_HEADING_HINTS)

            if line_has_education and not line_has_experience:
                continue
            if window_has_experience:
                context_years.extend(year_tokens)

        if not context_years:
            has_practical_hint = any(h in normalized for h in _EXPERIENCE_HEADING_HINTS)
            if not has_practical_hint:
                return None
            loose_years: list[int] = []
            for line in lines:
                year_tokens = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", line)]
                if not year_tokens:
                    continue
                if _looks_like_letter_spaced_text(line):
                    continue
                if any(h in line for h in _EDUCATION_HEADING_HINTS):
                    continue
                if any(k in line for k in ("high-school", "high school", "section")):
                    continue
                loose_years.extend(year_tokens)
            if not loose_years:
                return None
            lo = min(loose_years)
            hi = min(max(loose_years), current_year)
            if hi < lo:
                return None
            span = hi - lo
            return 1.0 if span <= 0 else round(float(span), 2)
        lo = min(context_years)
        hi = min(max(context_years), current_year)
        if hi < lo:
            return None
        span = hi - lo
        return 1.0 if span <= 0 else round(float(span), 2)

    ranges.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = []
    for s, e in ranges:
        if not merged or s > (merged[-1][1] + 1):
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    total_years = sum((e - s) for s, e in merged)
    return round(float(total_years), 2)

def detect_skills(text: str, known_skills: Iterable[str] | None = None) -> list[str]:
    if not known_skills:
        return []
    return [row["skill"] for row in detect_skills_with_confidence(text, known_skills)]

def _stage_error(stage: str, exc: Exception) -> str:
    return f"{stage}:{exc.__class__.__name__}"

def parse_cv_safe(
    file_bytes: bytes,
    filename: str,
    known_skills: Iterable[str] | None = None,
    min_confidence: float = 0.6,
    use_semantic: bool = False,
) -> dict[str, Any]:
    safe_filename = filename if isinstance(filename, str) else str(filename or "")
    safe_bytes: bytes
    if isinstance(file_bytes, bytes):
        safe_bytes = file_bytes
    elif isinstance(file_bytes, bytearray):
        safe_bytes = bytes(file_bytes)
    else:
        safe_bytes = b""

    result: dict[str, Any] = {
        "filename": safe_filename,
        "ok": True,
        "degraded": False,
        "errors": [],
        "warnings": [],
        "text_length": 0,
        "skills": [],
        "preview": "",
        "extracted_skills": [],
        "predicted_title": None,
        "predicted_experience_years": None,
    }

    if not isinstance(file_bytes, (bytes, bytearray)):
        result["degraded"] = True
        result["warnings"].append("invalid_file_bytes_type")

    skills_list = [
        str(s).strip()
        for s in (known_skills or [])
        if isinstance(s, str) and str(s).strip()
    ]
    if len(skills_list) > MAX_KNOWN_SKILLS:
        skills_list = skills_list[:MAX_KNOWN_SKILLS]
        result["degraded"] = True
        result["warnings"].append("known_skills_truncated")

    try:
        safe_min_conf = float(min_confidence)
    except Exception:
        safe_min_conf = 0.6
        result["degraded"] = True
        result["warnings"].append("invalid_min_confidence_defaulted")
    safe_min_conf = max(0.0, min(1.0, safe_min_conf))

    try:
        text = extract_text(safe_bytes, safe_filename)
    except Exception as exc:
        result["ok"] = False
        result["degraded"] = True
        result["errors"].append(_stage_error("extract_text", exc))
        return result

    if not isinstance(text, str):
        text = ""
        result["degraded"] = True
        result["warnings"].append("invalid_extracted_text_type")

    text = text or ""
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        result["degraded"] = True
        result["warnings"].append("text_truncated")

    result["text_length"] = len(text)
    result["preview"] = text[:200]

    if not text.strip():
        result["degraded"] = True
        result["warnings"].append("empty_text")
        return result

    try:
        skill_budget = DEFAULT_SKILL_TIME_BUDGET_SECONDS
        if len(text) > 80_000:
            skill_budget = 0.20
        start = time.perf_counter()
        rows = detect_skills_with_confidence(
            text=text,
            known_skills=skills_list,
            min_confidence=safe_min_conf,
            use_semantic=use_semantic,
            time_budget_seconds=skill_budget,
        )
        elapsed = time.perf_counter() - start
        budget_hit = bool(skill_budget and elapsed >= (skill_budget * 0.98))

        if not isinstance(rows, list):
            rows = []
            result["degraded"] = True
            result["warnings"].append("invalid_skills_output")

        result["extracted_skills"] = rows
        result["skills"] = [
            str(r.get("skill")).strip()
            for r in rows
            if isinstance(r, dict) and r.get("skill")
        ]
        if not result["skills"]:
            result["warnings"].append("no_skills_detected")
        if budget_hit:
            result["warnings"].append("skills_time_budget_hit")
            if not result["skills"]:
                # Budget pressure is only quality-degrading when extraction returns no skills.
                result["degraded"] = True
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("detect_skills_with_confidence", exc))
        try:
            legacy = detect_skills(text=text, known_skills=skills_list)
            result["skills"] = [str(s).strip() for s in legacy if isinstance(s, str) and str(s).strip()]
            result["extracted_skills"] = [
                {"skill": s, "confidence": 0.6, "source": "legacy"} for s in result["skills"]
            ]
            result["warnings"].append("legacy_skill_fallback")
        except Exception as exc2:
            result["errors"].append(_stage_error("detect_skills_legacy", exc2))

    try:
        result["predicted_title"] = detect_title(text)
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("detect_title", exc))

    try:
        result["predicted_experience_years"] = detect_experience_years(text)
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("detect_experience_years", exc))

    return result
