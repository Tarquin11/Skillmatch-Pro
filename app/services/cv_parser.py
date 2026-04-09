import io
import re
import math
import time
import unicodedata
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable
import numpy as np
import pdfplumber
from docx import Document
from app.ai.confidence_calibration import apply_platt_on_unit_interval, load_platt_params
from app.ai.skill_canonicalization import canonicalize_skill
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_SKILL_HEADING_HINTS = (
    "skills",
    "technical skills",
    "technologies",
    "technology",
    "tech stack",
    "stack",
    "tools",
    "competencies",
    "core skills",
    "languages",
    "language",
    "software",
    "platforms",
    "frameworks",
    "libraries",
    "expertise",
    "proficiencies",
    "certifications",
    "methodologies",
    "competences",
    "compétences",
    "competences techniques",
    "compétences techniques",
    "outils",
    "langues",
    "framework",
    "bibliotheques",
    "bibliothèques",
)
_LANGUAGE_HEADING_HINTS = ("language", "languages", "langue", "langues", "idioma", "idiomas")
_DURATION_PHRASE_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:months?|years?|yrs?)$", re.I)
WORD_RE = re.compile(r"[a-z0-9+.#/\-]+")
_EXPERIENCE_YEARS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|ans)\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(present|current|now|19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_MONTH_NAME_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
_YEAR_TOKEN_RE = re.compile(r"\b(19|20)\d{2}\b")
_LEADING_DATE_RANGE_RE = re.compile(
    r"^\s*\d{1,2}\s+[A-Za-z]+\s*[-–—]\s*\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?\s+",
    re.IGNORECASE,
)
_OPEN_VOCAB_BOILERPLATE = (
    "implementation of",
    "development of",
    "contribution to",
    "application for",
    "system to provide",
    "functionality to calculate",
    "functionality to",
    "tracking public",
    "regional transport",
    "transport society",
    "mobile app developement",
    "mobile app development",
    "pentesting backend",
    "based bus",
    "bus geolocation",
    "mise en place de",
    "developpement de",
    "développement de",
    "contribution au",
    "contribution a",
    "application de suivi",
    "real-time positioning",
    "positioning system",
)

_LANGUAGE_ALIAS_MAP = {
    "english": "english",
    "anglais": "english",
    "french": "french",
    "francais": "french",
    "français": "french",
    "arabic": "arabic",
    "arabe": "arabic",
    "swedish": "swedish",
    "suedois": "swedish",
    "suedois": "swedish",
    "suédois": "swedish",
    "spanish": "spanish",
    "espagnol": "spanish",
    "german": "german",
    "allemand": "german",
    "italian": "italian",
    "italien": "italian",
    "portuguese": "portuguese",
    "portugais": "portuguese",
    "dutch": "dutch",
    "neerlandais": "dutch",
    "néerlandais": "dutch",
    "russian": "russian",
    "russe": "russian",
    "chinese": "chinese",
    "chinois": "chinese",
    "japanese": "japanese",
    "japonais": "japanese",
    "turkish": "turkish",
    "turc": "turkish",
}
_CEFR_RE = re.compile(r"\b(A1|A2|B1|B2|C1|C2)\b", re.IGNORECASE)
_LANGUAGE_ALIAS_MAP_NORM = {
    re.sub(r"[^a-z0-9\s]", " ", unicodedata.normalize("NFKD", k).encode("ascii", "ignore").decode("ascii").lower()).strip(): v
    for k, v in _LANGUAGE_ALIAS_MAP.items()
}

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
    "charge de projets",
    "chargé de projets",
    "chef de projet",
    "project manager",
    "gestionnaire",
    "coordinateur",
    "coordonnateur",
)

# One-line section titles (normalized). Avoid treating skill bullets like "Python" as headings.
_SINGLE_WORD_RESUME_HEADINGS = frozenset(
    {
        "skills",
        "tools",
        "languages",
        "language",
        "education",
        "experience",
        "projects",
        "project",
        "summary",
        "profile",
        "overview",
        "contact",
        "references",
        "certifications",
        "awards",
        "publications",
        "patents",
        "volunteering",
        "volunteer",
        "hobbies",
        "interests",
        "activities",
        "employment",
        "internships",
        "internship",
        "objective",
        "highlights",
        "achievements",
        "strengths",
        "competences",
        "compétences",
        "outils",
        "langues",
        "formations",
        "formation",
    }
)

_MULTI_WORD_RESUME_HEADINGS = (
    "technical skills",
    "core skills",
    "soft skills",
    "hard skills",
    "tech stack",
    "technical stack",
    "work experience",
    "professional experience",
    "professional summary",
    "personal projects",
    "academic projects",
    "professional projects",
    "career objective",
    "education and training",
    "additional information",
    "competences techniques",
    "compétences techniques",
    "experience professionnelle",
    "expérience professionnelle",
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
DEFAULT_SKILL_TIME_BUDGET_SECONDS = 0.75
LAYOUT_SHORT_TEXT_FILE_BYTES_THRESHOLD = 30_000
LAYOUT_SHORT_TEXT_NONSPACE_THRESHOLD = 120
LAYOUT_COLUMN_MIN_LINES = 24
LAYOUT_COLUMN_SHORT_LINE_WORDS = 3
LAYOUT_COLUMN_SHORT_LINE_RATIO = 0.72
LAYOUT_LETTER_SPACED_MIN_LINES = 10
LAYOUT_LETTER_SPACED_RATIO = 0.35


def _is_skill_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _SKILL_HEADING_HINTS)


def _is_language_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _LANGUAGE_HEADING_HINTS)

def _is_noise_skill_phrase(phrase: str) -> bool:
    p = _normalize_text(phrase)
    return bool(_DURATION_PHRASE_RE.fullmatch(p))

def _clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _is_noise_line(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _post_ocr_text_normalize(text: str) -> str:
    """
    Light cleanup for PDF/OCR-style spacing (Latin scripts).
    NFKD + strip combining marks approximates unidecode for accents without an extra dependency.
    Preserves line breaks so section detection still works.
    """
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    out_lines: list[str] = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"\b([A-Z])\s+([a-z])", r"\1\2", line)
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def extract_text(file_bytes, filename):
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
            text = _post_ocr_text_normalize(text)
            return _clean_extracted_text(text)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        text = _post_ocr_text_normalize(text)
        return _clean_extracted_text(text)
    if name.endswith(".txt"):
        raw = (file_bytes or b"").decode("utf-8", errors="replace")
        text = _post_ocr_text_normalize(raw)
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
    # "T O O L S" is 5 spaced letters; "S K I L L S" is 6 — both are common in styled CV PDFs.
    if len(alpha_parts) < 5:
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


def _matches_resume_section_heading(key: str, words: list[str]) -> bool:
    """True if this short line is a typical CV section title, not a skill/tool bullet."""
    if not key or not words:
        return False
    if key in _SINGLE_WORD_RESUME_HEADINGS:
        return True
    for phrase in _MULTI_WORD_RESUME_HEADINGS:
        if key == phrase or key.startswith(phrase + " ") or key.endswith(" " + phrase):
            return True
    if len(words) >= 2:
        last = _normalize_text(words[-1])
        if last in (
            "skills",
            "tools",
            "technologies",
            "languages",
            "stack",
            "competencies",
            "expertise",
            "proficiencies",
        ):
            return True
        first = _normalize_text(words[0])
        if first in (
            "technical",
            "core",
            "personal",
            "professional",
            "work",
            "soft",
            "hard",
            "key",
            "relevant",
        ) and len(words) <= 5:
            return True
    return False


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
    if _looks_like_letter_spaced_text(line):
        line = _collapse_letter_spaced_text(line)
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
    has_colon = raw_line.rstrip().endswith(":")
    key = _normalize_text(trimmed)
    section_like = _matches_resume_section_heading(key, words)

    # Short all-caps tokens (SQL, AWS, GCP) are usually skills, not section titles.
    short_all_caps_not_section = (
        not has_colon
        and not section_like
        and len(words) == 1
        and len(trimmed) <= 5
        and trimmed.isalpha()
        and trimmed.isupper()
    )

    # Do not use loose "Title Case" alone — it mis-tags bullets like "Python" or "Web Exploitation".
    is_heading = has_colon or section_like or (upper_ratio >= 0.6 and not short_all_caps_not_section)
    if not is_heading:
        return False, "", 0.0

    weight = 1.1 if has_colon or upper_ratio >= 0.75 or section_like else 1.0
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
            parts = re.split(r"[,\|;/&]+", line)
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
# Skill vectors keyed by sorted known_skills tuple hash — reuse across CVs with same catalog slice.
_EMBED_CACHE: dict[int, tuple[np.ndarray, list[str]]] = {}
_HF_SKILL_NER_PIPE: Callable[[str], list[dict[str, Any]]] | None = None
_HF_SKILL_NER_LOAD_ATTEMPTED = False

def _get_embedder() -> EmbeddingService | None:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        _EMBEDDER = EmbeddingService()
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER


def _get_hf_skill_ner_pipeline() -> Callable[[str], list[dict[str, Any]]] | None:
    """Lazy-load HF token-classification model for optional NER spans."""
    global _HF_SKILL_NER_PIPE
    global _HF_SKILL_NER_LOAD_ATTEMPTED
    if _HF_SKILL_NER_PIPE is not None:
        return _HF_SKILL_NER_PIPE
    if _HF_SKILL_NER_LOAD_ATTEMPTED:
        return None
    _HF_SKILL_NER_LOAD_ATTEMPTED = True
    try:
        from transformers import pipeline

        _HF_SKILL_NER_PIPE = pipeline(
            "token-classification",
            model="jjzha/escoxlmr_skill_extraction",
            aggregation_strategy="simple",
        )
    except Exception:
        _HF_SKILL_NER_PIPE = None
    return _HF_SKILL_NER_PIPE


def _extract_hf_ner_spans(text: str, max_spans: int = 200) -> list[str]:
    ner = _get_hf_skill_ner_pipeline()
    if ner is None:
        return []
    safe_text = (text or "").strip()[:24000]
    if not safe_text:
        return []
    try:
        rows = ner(safe_text)
    except Exception:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        span = str(row.get("word") or "").strip()
        if not span:
            continue
        # Keep spans compact; parser does semantic remapping later.
        span = re.sub(r"\s+", " ", span).strip(" -,:.;")
        if not span:
            continue
        key = _normalize_text(span)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(span)
        if len(out) >= max_spans:
            break
    return out

SEMANTIC_AUGMENT_MIN_SIMILARITY = 0.80
SEMANTIC_AUGMENT_MAX_SPANS = 72
SEMANTIC_AUGMENT_TOP_CANDIDATES = 3

# Negation in the same line/span as a semantic candidate → reject (avoid "no experience in X").
_SPAN_NEGATION_RE = re.compile(
    r"(?i)"
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\blacking\b|\black\s+of\b|\bneither\b|\bnor\b|"
    r"\bno\s+experience\b|\bwithout\s+experience\b|"
    r"\bsans\b|\baucune\b|\baucun\b|\bjamais\b|"
    r"\bpas\s+de\b|\bpas\s+d['\u2019]|"
    r"\bni\s+l['\u2019]?\s*experience\b|\bzero\s+experience\b|\bz[eé]ro\s+exp[eé]rience\b)"
)

# Weak / hedged wording in evidence → downweight confidence (adversarial phrasing).
_NEGATION_PREFIX_RE = re.compile(
    r"(?i)"
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\blacking\b|\black\s+of\b|"
    r"\bsans\b|\baucun(?:e)?\b|\bjamais\b|\bpas\s+de\b|\bpas\s+d['\u2019])"
)
_NEGATION_SUFFIX_RE = re.compile(r"(?i)^\s*(?:not|never|jamais|absent|missing|manquant(?:e)?)\b")
_FR_NE_PAS_RE = re.compile(
    r"(?i)"
    r"(?:\bne\b(?:\s+\w+){0,7}\s+\bpas\b|\bn['\u2019]\w*(?:\s+\w+){0,7}\s+\bpas\b)"
)
_SENTENCE_BREAK_RE = re.compile(r"[.!?;:]")
_NEGATION_LINK_TAIL_RE = re.compile(
    r"(?i)(?:\bwith\b|\bin\b|\bon\b|\bfor\b|\bto\b|\bde\b|\bd['\u2019]?\b|\ben\b|\bsur\b|\bavec\b|\bsans\b|\bwithout\b)\s*$"
)

_WEAK_HEDGE_RE = re.compile(
    r"(?i)\b("
    r"basic|familiar|familiarity|exposed|exposure|"
    r"notions?|awareness|aware\s+of|"
    r"limited|rough|rudimentary|superficial|"
    r"surface\s+level|introduction\s+to|intro\s+to|"
    r"bases\s+en|sensibilisation|decouverte|découverte|vue\s+d"
    r")\b"
)


def _span_text_negated(span: str) -> bool:
    if not (span or "").strip():
        return False
    norm = _normalize_for_pattern(span)
    return bool(_SPAN_NEGATION_RE.search(norm))


def _skill_mention_patterns(skill: str) -> list[str]:
    canonical = canonicalize_skill(skill or "")
    if not canonical:
        return []
    patterns: list[str] = []
    literal = re.escape(_normalize_for_pattern(canonical))
    if literal:
        patterns.append(rf"\b{literal}\b")
    for rx, can in _SKILL_SENTENCE_PATTERNS.items():
        if canonicalize_skill(can) == canonical:
            patterns.append(rx)
    return patterns


def _segment_tail_for_scope(prefix: str) -> str:
    local = (prefix or "")[-160:]
    matches = list(_SENTENCE_BREAK_RE.finditer(local))
    if matches:
        local = local[matches[-1].end() :]
    return local.strip()


def _segment_head_for_scope(suffix: str) -> str:
    local = (suffix or "")[:100]
    matches = list(_SENTENCE_BREAK_RE.finditer(local))
    if matches:
        local = local[: matches[0].start()]
    return local.strip()


def _mention_negated_in_context(skill: str, context_text: str) -> bool:
    norm = _normalize_for_pattern(context_text or "")
    if not norm:
        return False
    for patt in _skill_mention_patterns(skill):
        try:
            for m in re.finditer(patt, norm):
                prefix = _segment_tail_for_scope(norm[max(0, m.start() - 180) : m.start()])
                suffix = _segment_head_for_scope(norm[m.end() : m.end() + 80])
                if _NEGATION_PREFIX_RE.search(prefix):
                    return True
                if _FR_NE_PAS_RE.search(prefix):
                    return True
                if _NEGATION_SUFFIX_RE.search(suffix):
                    return True
        except re.error:
            continue
    return False


def _skill_negated_in_text(skill: str, text: str, *, window_lines: int = 1) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
    if not lines:
        return False

    def _prev_line_links_to_current(prev: str) -> bool:
        norm_prev = _normalize_for_pattern(prev or "").strip()
        if not norm_prev:
            return False
        if _SENTENCE_BREAK_RE.search(norm_prev[-1:]):
            return False
        if _NEGATION_LINK_TAIL_RE.search(norm_prev):
            return True
        return False

    neg_hits = 0
    pos_hits = 0
    for i, line in enumerate(lines):
        if not _skill_line_hit(skill, line):
            continue
        linked_prev: list[str] = []
        for off in range(1, int(max(0, window_lines)) + 1):
            j = i - off
            if j < 0:
                break
            if _prev_line_links_to_current(lines[j]):
                linked_prev.insert(0, lines[j])
            else:
                break
        window = "\n".join(linked_prev + [line])
        if _mention_negated_in_context(skill, window):
            neg_hits += 1
        else:
            pos_hits += 1
    return neg_hits > 0 and pos_hits == 0


def filter_negated_skill_rows(
    rows: list[dict[str, Any]],
    text: str,
    *,
    window_lines: int = 1,
) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    dropped: list[str] = []
    seen_drop: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("skill"):
            continue
        skill = str(row.get("skill", "")).strip()
        if _skill_negated_in_text(skill, text, window_lines=window_lines):
            can = canonicalize_skill(skill)
            if can and can not in seen_drop:
                seen_drop.add(can)
                dropped.append(can)
            continue
        out.append(row)
    return out, sorted(dropped)


def _evidence_weak_hedge(blob: str) -> bool:
    if not (blob or "").strip():
        return False
    return bool(_WEAK_HEDGE_RE.search(_normalize_for_pattern(blob)))


def apply_weak_hedge_penalty_to_rows(rows: list[dict[str, Any]], penalty: float = 0.07) -> None:
    """Reduce confidence when evidence uses weak / introductory phrasing."""
    for r in rows:
        if not isinstance(r, dict):
            continue
        blob = " ".join(str(x) for x in (r.get("evidence") or []) if x)
        if not _evidence_weak_hedge(blob):
            continue
        try:
            c = float(r.get("confidence", 0))
        except Exception:
            c = 0.0
        r["confidence"] = round(max(0.52, c - penalty), 2)


def attach_confidence_normalized(rows: list[dict[str, Any]]) -> None:
    """Per-CV relative score: confidence / max(confidence) in this extraction (for fair ranking).

    Max-skill is always 1.0. For softmax-style pooling across CVs, compute downstream from
    these raw confidences or from stored `confidence` values.
    """
    confs: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            confs.append(float(r.get("confidence", 0)))
        except Exception:
            pass
    mx = max(confs) if confs else 1.0
    if mx <= 0:
        mx = 1.0
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            c = float(r.get("confidence", 0))
        except Exception:
            c = 0.0
        r["confidence_normalized"] = round(min(1.0, max(0.0, c / mx)), 4)


def _spans_for_semantic_augment(text: str) -> list[str]:
    """Non-overlapping substantive lines; each span is evidence for any suggestion."""
    spans: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines()[:MAX_LINES]:
        line = raw.strip()
        if not line or _is_noise_line(line):
            continue
        for p in BULLET_PREFIXES:
            if line.startswith(p):
                line = line[len(p) :].strip()
                break
        if len(line) < 12:
            continue
        line = re.sub(r"\s+", " ", line)[:240]
        key = _normalize_text(line)
        if not key or key in seen:
            continue
        seen.add(key)
        spans.append(line)
        if len(spans) >= SEMANTIC_AUGMENT_MAX_SPANS:
            break
    return spans


def _similarity_to_augment_base_confidence(sim: float) -> float:
    """Keep suggested scores below strong rule-based hits (augmentation, not replacement)."""
    lo = SEMANTIC_AUGMENT_MIN_SIMILARITY
    t = (float(sim) - lo) / max(1e-6, 1.0 - lo)
    t = max(0.0, min(1.0, t))
    return 0.60 + 0.22 * t


def augment_skills_semantically_gated(
    text: str,
    known_skills: list[str],
    existing_skill_keys: set[str],
    *,
    min_confidence: float = 0.6,
    min_similarity: float = SEMANTIC_AUGMENT_MIN_SIMILARITY,
    max_additions: int = 15,
    time_budget_seconds: float | None = 0.45,
) -> list[dict[str, Any]]:
    """
    Gated semantic augmenter: embed line spans vs catalog, add skills only when
    similarity is high, the skill is not already detected, and a text span exists
    as evidence. Does not modify or replace rule/pattern rows.
    """
    spans = _spans_for_semantic_augment(text)
    if not spans or not known_skills:
        return []
    skill_vectors, skill_names = _get_skill_embeddings(known_skills)
    if skill_vectors is None or not skill_names:
        return []
    embedder = _get_embedder()
    if not embedder:
        return []
    deadline = None
    if time_budget_seconds and time_budget_seconds > 0:
        deadline = time.perf_counter() + float(time_budget_seconds)

    def _over_time() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    try:
        span_vectors = np.asarray(embedder.generate_embeddings(spans), dtype=np.float32)
    except Exception:
        return []
    if span_vectors.size == 0:
        return []

    sims = span_vectors @ skill_vectors.T
    # skill_key -> (best_sim, evidence_span, display_name)
    best_for_skill: dict[str, tuple[float, str, str]] = {}

    for i, span in enumerate(spans):
        if _over_time():
            break
        row = sims[i]
        n_sk = int(row.shape[0])
        kk = min(SEMANTIC_AUGMENT_TOP_CANDIDATES, n_sk)
        idxs = np.argpartition(-row, kk - 1)[:kk]
        idxs = sorted([int(j) for j in idxs], key=lambda j: -float(row[j]))
        for j in idxs:
            sim = float(row[j])
            if sim < min_similarity:
                continue
            if _span_text_negated(span):
                continue
            display = skill_names[j]
            canonical = canonicalize_skill(display)
            sk = _skill_key(canonical)
            if not sk or sk in existing_skill_keys:
                continue
            prev = best_for_skill.get(sk)
            if prev is None or sim > prev[0]:
                best_for_skill[sk] = (sim, span, canonical)

    ranked = sorted(best_for_skill.values(), key=lambda t: -t[0])[:max_additions]
    out: list[dict[str, Any]] = []
    for sim, span, canonical in ranked:
        if _span_text_negated(span):
            continue
        ev_span = re.sub(r"\s+", " ", span).strip()
        if len(ev_span) > 180:
            ev_span = ev_span[:177] + "..."
        conf = _similarity_to_augment_base_confidence(sim)
        if conf < min_confidence:
            continue
        out.append(
            {
                "skill": canonical,
                "confidence": round(conf, 2),
                "source": "semantic_augment",
                "evidence": [ev_span],
                "_conf_channels": {"semantic_augment"},
            }
        )
    return out


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
        "ner_span": 0.94,
        "legacy": 0.88,
    }.get(source_key, 0.92)

    sec = max(0.75, min(1.05, section_weight))
    adjusted = raw_conf * source_weight * sec
    adjusted = max(0.001, min(0.999, adjusted))
    temp = 1.15 if source_key in ("fuzzy","semantic") else 1.0
    logit = math.log(adjusted / (1 - adjusted))
    calibrated = 1.0 / (1.0 + math.exp(-logit / temp))
    return max(0.01, min(0.99, calibrated))


def _extract_cefr_level(raw: str) -> str | None:
    m = _CEFR_RE.search(raw or "")
    if not m:
        return None
    return m.group(1).upper()

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

        calibrated = _calibrate_confidence(best_conf, best_source, best_section_weight)
        if calibrated >= min_confidence:
            hits.append(
                {
                    "skill": canonical,
                    "confidence": round(calibrated, 2),
                    "source": best_source,
                }
            )
    hits.sort(key=lambda row: (-float(row["confidence"]), str(row["skill"])))
    return hits


_GENERIC_OPEN_VOCAB_DROP = frozenset(
    {
        "and",
        "or",
        "etc",
        "other",
        "others",
        "various",
        "multiple",
        "strong",
        "basic",
        "advanced",
        "none",
        "n/a",
        "na",
        "some",
        "many",
    }
)
_SOFT_SKILL_ALLOWLIST = frozenset(
    {
        "leadership",
        "communication",
        "travail dequipe",
        "teamwork",
        "relation client",
        "customer relationship",
        "autonomie",
        "autonomy",
        "capacite dadaptation",
        "adaptability",
        "gestion de projets",
        "project management",
    }
)
_SHORT_GENERIC_DROP = frozenset(
    {
        "client",
        "clients",
        "organisation",
        "organisations",
        "organization",
        "organizations",
        "equipe",
        "equipes",
        "team",
        "teams",
    }
)
_SOFT_SKILL_HINTS: dict[str, str] = {
    "leadership": "leadership",
    "communication": "communication",
    "relation client": "customer relationship",
    "customer relationship": "customer relationship",
    "gestion de projet": "project management",
    "gestion de projets": "project management",
    "project management": "project management",
    "gestion du budget": "budget management",
    "budget": "budget management",
    "gestion des risques": "risk management",
    "risk management": "risk management",
    "coordination": "coordination",
    "travail dequipe": "teamwork",
    "teamwork": "teamwork",
    "capacite dadaptation": "adaptability",
    "adaptability": "adaptability",
    "planifier les projets": "planning",
    "planification": "planning",
    "calendrier": "scheduling",
    "echeancier": "scheduling",
    "echeance": "scheduling",
    "risk": "risk management",
    "risques": "risk management",
    "systemes informatiques": "it systems",
    "systeme informatique": "it systems",
    "coordonner": "coordination",
    "coordonner": "coordination",
    "projects": "project management",
    "projets": "project management",
}
_ACTION_VERB_PREFIXES = (
    "gerer ",
    "gérer ",
    "transmettre ",
    "connaitre ",
    "connaître ",
    "savoir ",
    "planifier ",
    "coordonner ",
    "definir ",
    "définir ",
    "anticiper ",
    "analyser ",
    "apporter ",
    "aider ",
    "accompagner ",
    "concevoir ",
    "organiser ",
    "manage ",
    "define ",
    "coordinate ",
    "plan ",
    "analyze ",
    "help ",
    "support ",
)
_LEADING_DETERMINERS = ("des ", "les ", "la ", "le ", "du ", "de ", "d ")
_LANGUAGE_PROFICIENCY_HINTS = (
    "langue",
    "langues",
    "language",
    "languages",
    "native",
    "mother tongue",
    "courant",
    "intermediaire",
    "intermediate",
    "fluent",
    "niveau",
    "level",
)
_LANGUAGE_FALSE_POSITIVE_HINTS = (
    "baccalaureat",
    "baccalauréat",
    "lycee",
    "lycée",
    "universite",
    "université",
    "paris",
    "montreal",
    "montréal",
)
_NOISE_LINE_HINTS = (
    "modeles-de-cv",
    "modeles de cv",
    "copyright",
    "all rights reserved",
    "linkedin.com/",
    "http://",
    "https://",
)
_SKILL_SENTENCE_PATTERNS: dict[str, str] = {
    r"\b(gestion de projet|gerer .*projet|gérer .*projet|chef de projet)\b": "project management",
    r"\b(planifier .*projet|planification|echeancier|échéancier|calendrier)\b": "planning",
    r"\b(gestion des risques|risques|risk management)\b": "risk management",
    r"\b(coordonner .*equipe|coordonner .*équipe|coordination)\b": "coordination",
    r"\b(budget|gestion du budget|budgeting)\b": "budget management",
    r"\b(systemes informatiques|systèmes informatiques|it systems)\b": "it systems",
    r"\b(travail d equipe|travail d'équipe|teamwork)\b": "teamwork",
    r"\b(adaptation|adaptabilite|adaptabilité|adaptability)\b": "adaptability",
    r"\b(communication)\b": "communication",
    r"\b(leadership)\b": "leadership",
    r"\b(relation client|client relationship|customer relationship)\b": "customer relationship",
    r"\b(organisation|organization)\b": "organization",
}
_SKILL_CATEGORY_MAP: dict[str, str] = {
    "project management": "management",
    "planning": "management",
    "scheduling": "management",
    "coordination": "management",
    "budget management": "business",
    "risk management": "management",
    "customer relationship": "business",
    "communication": "soft-skills",
    "leadership": "soft-skills",
    "organization": "business",
    "it systems": "technical",
    "python": "technical",
    "sql": "technical",
    "docker": "technical",
    "adaptability": "soft-skills",
    "teamwork": "soft-skills",
}
_SKILL_HIERARCHY_MAP: dict[str, list[str]] = {
    "project management": ["planning", "scheduling", "budget management", "risk management", "coordination"],
}


def _strip_parenthetical_qualifiers(phrase: str) -> str:
    p = (phrase or "").strip()
    p = re.sub(r"\s*\([^)]*\)\s*$", "", p).strip()
    p = re.sub(r"\s*\[[^\]]*\]\s*$", "", p).strip()
    return p


def _strip_open_vocab_leading_junk(phrase: str) -> str:
    """Remove common PDF-merge prefixes (date ranges) before skill parsing."""
    s = (phrase or "").strip()
    s = _LEADING_DATE_RANGE_RE.sub("", s).strip()
    s = re.sub(r"^\d{4}\s+", "", s).strip()
    s = re.sub(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\s+", "", s).strip()
    return s


def _open_vocab_looks_like_noise_sentence(low: str) -> bool:
    """Reject project descriptions and CV narrative merged into one 'skill' line."""
    if _MONTH_NAME_RE.search(low) and _YEAR_TOKEN_RE.search(low):
        return True
    if re.search(r"\b\d{1,2}\s*[-–—/]\s*\d{1,2}\s+\w+\s+\d{4}\b", low):
        return True
    for frag in _OPEN_VOCAB_BOILERPLATE:
        if frag in low:
            return True
    if low.count(".") >= 1 and len(low) > 35:
        return True
    padded = f" {low} "
    stop_hits = sum(
        1
        for x in (
            " the ",
            " a ",
            " an ",
            " to ",
            " for ",
            " of ",
            " with ",
            " in ",
            " on ",
            " from ",
            " that ",
            " le ",
            " la ",
            " les ",
            " des ",
            " de ",
            " du ",
            " pour ",
            " avec ",
            " dans ",
            " sur ",
        )
        if x in padded
    )
    if stop_hits >= 3 and len(low.split()) >= 6:
        return True
    return False


def _is_noise_line(raw: str) -> bool:
    low = _normalize_text(raw)
    if not low:
        return True
    return any(h in low for h in _NOISE_LINE_HINTS)


def _normalize_for_pattern(text: str) -> str:
    low = _normalize_text(text)
    return unicodedata.normalize("NFKD", low).encode("ascii", "ignore").decode("ascii")


def _layout_quality_assessment(
    *,
    text: str,
    filename: str,
    file_bytes_len: int,
) -> tuple[bool, list[str]]:
    """Heuristic extraction quality checks for PDF/Word layout degradation.

    Returns `(degraded, warnings)` while keeping logic conservative:
    only trigger severe checks for realistically large source files.
    """
    warnings: list[str] = []
    degraded = False

    name = (filename or "").lower()
    is_layout_sensitive = name.endswith(".pdf") or name.endswith(".docx") or name.endswith(".doc")
    if not is_layout_sensitive:
        return False, warnings

    nonspace_len = len(re.sub(r"\s+", "", text or ""))
    if (
        file_bytes_len >= LAYOUT_SHORT_TEXT_FILE_BYTES_THRESHOLD
        and nonspace_len < LAYOUT_SHORT_TEXT_NONSPACE_THRESHOLD
    ):
        warnings.append("extraction_suspect_too_short")
        degraded = True

    lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
    if len(lines) >= LAYOUT_COLUMN_MIN_LINES:
        word_counts = [max(1, len(_tokenize(_normalize_for_pattern(ln)))) for ln in lines]
        short_ratio = sum(1 for c in word_counts if c <= LAYOUT_COLUMN_SHORT_LINE_WORDS) / max(1, len(word_counts))
        letter_spaced_ratio = sum(1 for ln in lines if _looks_like_letter_spaced_text(ln)) / max(1, len(lines))

        if short_ratio >= LAYOUT_COLUMN_SHORT_LINE_RATIO:
            warnings.append("layout_column_fragmentation_suspected")
            degraded = True
        if len(lines) >= LAYOUT_LETTER_SPACED_MIN_LINES and letter_spaced_ratio >= LAYOUT_LETTER_SPACED_RATIO:
            warnings.append("ocr_like_spacing_detected")
            degraded = True

    if degraded:
        warnings.append("ocr_or_table_extraction_recommended")

    # Keep stable order and no duplicates.
    return degraded, list(dict.fromkeys(warnings))


def _is_bullet_line(raw: str) -> bool:
    line = (raw or "").strip()
    return line.startswith(BULLET_PREFIXES)


def extract_sentence_skill_rows(
    text: str,
    min_confidence: float = 0.6,
    max_rows: int = 40,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    Sentence-level extraction independent of strict section boundaries.
    Prioritizes bullet lines and action/object skill phrases.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    lines = [(ln or "").strip() for ln in (text or "").splitlines()[:MAX_LINES]]
    for line in lines:
        if not line or _is_noise_line(line):
            continue
        norm = _normalize_for_pattern(line)
        if not norm:
            continue
        bullet = _is_bullet_line(line)
        for patt, canonical in _SKILL_SENTENCE_PATTERNS.items():
            if not re.search(patt, norm):
                continue
            sk = _skill_key(canonical)
            if not sk or sk in seen:
                continue
            seen.add(sk)
            base = 0.76 if bullet else 0.70
            conf = round(max(float(min_confidence), min(0.90, base + (0.03 if len(canonical.split()) >= 2 else 0.0))), 2)
            row = {
                "skill": canonical,
                "confidence": conf,
                "source": "sentence_bullet" if bullet else "sentence_text",
            }
            out.append(row)
            if debug:
                logger.debug("sentence_skill_detected sentence=%r skill=%s source=%s", line, canonical, row["source"])
            if len(out) >= max_rows:
                return out
    out.sort(key=lambda row: (-float(row["confidence"]), str(row["skill"])))
    return out


def _open_vocab_phrase_ok(raw: str, *, language_section: bool = False) -> bool:
    phrase = _strip_open_vocab_leading_junk(raw)
    phrase = _strip_parenthetical_qualifiers(phrase)
    phrase = phrase.strip()
    if not phrase:
        return False
    display = canonicalize_skill(phrase).strip()
    if not display:
        return False
    if _is_noise_skill_phrase(phrase):
        return False
    low = _normalize_text(display)
    low_core = re.sub(r"[^a-z0-9\s]", "", low).strip()
    if not low_core or low_core in _GENERIC_OPEN_VOCAB_DROP:
        return False
    soft_key = low_core
    if soft_key in _SOFT_SKILL_ALLOWLIST:
        return True
    if soft_key in _SHORT_GENERIC_DROP:
        return False
    if any(low.startswith(p) for p in _ACTION_VERB_PREFIXES):
        return False
    words = low_core.split()
    if any(low_core.startswith(prefix) for prefix in _LEADING_DETERMINERS) and len(words) <= 4:
        return False
    if len(words) <= 2 and all(w in _GENERIC_OPEN_VOCAB_DROP for w in words):
        return False
    if len(low_core) < 2:
        return False
    max_chars = 72 if language_section else 56
    if len(display) > max_chars:
        return False
    max_words = 5 if language_section else 6
    if len(words) > max_words:
        return False
    if _open_vocab_looks_like_noise_sentence(low):
        return False
    return True


def _display_open_vocab_skill(raw: str) -> str:
    phrase = _strip_open_vocab_leading_junk(raw)
    phrase = _strip_parenthetical_qualifiers(phrase)
    d = canonicalize_skill(phrase).strip()
    return (d or phrase.strip())[:120]


def _normalize_language_name(raw: str) -> str:
    low = _normalize_text(raw)
    low = unicodedata.normalize("NFKD", low).encode("ascii", "ignore").decode("ascii")
    low = re.sub(r"[^a-z0-9\s]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return _LANGUAGE_ALIAS_MAP_NORM.get(low, "")


def extract_language_details_from_sections(text: str, max_languages: int = 12) -> list[dict[str, str]]:
    sections, _ = _extract_sections(text)
    section_phrases = _extract_section_phrases(sections)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for phrase, section in section_phrases:
        if not _is_language_heading(section):
            continue
        cleaned = _strip_open_vocab_leading_junk(phrase)
        cefr = _extract_cefr_level(cleaned)
        cleaned = _strip_parenthetical_qualifiers(cleaned)
        lang = _normalize_language_name(cleaned)
        if not lang or lang in seen:
            continue
        seen.add(lang)
        item: dict[str, str] = {
            "language": lang,
            "source": f"language_section:{re.sub(r'[^a-z0-9]+', '_', _normalize_text(section)).strip('_') or 'section'}",
        }
        if cefr:
            item["level"] = cefr
        out.append(item)
        if len(out) >= max_languages:
            break
    return out


def extract_languages_from_sections(text: str, max_languages: int = 12) -> list[str]:
    return [row.get("language", "") for row in extract_language_details_from_sections(text, max_languages=max_languages) if row.get("language")]


def extract_language_details_from_anywhere(
    text: str,
    existing_languages: set[str] | None = None,
    max_languages: int = 12,
) -> list[dict[str, str]]:
    """
    Fallback when PDF layout breaks language headings.
    Scans raw lines for known language mentions + optional CEFR levels.
    """
    out: list[dict[str, str]] = []
    seen = set(existing_languages or set())
    for raw_line in (text or "").splitlines()[:MAX_LINES]:
        line = raw_line.strip()
        if not line:
            continue
        norm = _normalize_text(line)
        norm = unicodedata.normalize("NFKD", norm).encode("ascii", "ignore").decode("ascii")
        norm = re.sub(r"[^a-z0-9\s]", " ", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        if not norm:
            continue
        if any(h in norm for h in _LANGUAGE_FALSE_POSITIVE_HINTS):
            continue
        cefr_matches = [(m.start(), m.group(1).upper()) for m in _CEFR_RE.finditer(norm)]
        lang_mentions = 0
        for key in _LANGUAGE_ALIAS_MAP_NORM.keys():
            if key and re.search(rf"\b{re.escape(key)}\b", norm):
                lang_mentions += 1
        has_proficiency_hint = any(h in norm for h in _LANGUAGE_PROFICIENCY_HINTS)
        if not (cefr_matches or lang_mentions >= 2 or has_proficiency_hint):
            continue
        hits: list[tuple[int, int, str]] = []
        for key, canonical in _LANGUAGE_ALIAS_MAP_NORM.items():
            if not key or canonical in seen:
                continue
            for m in re.finditer(rf"\b{re.escape(key)}\b", norm):
                hits.append((m.start(), m.end(), canonical))
        hits.sort(key=lambda h: h[0])
        deduped: list[tuple[int, int, str]] = []
        dup_lang: set[str] = set()
        for s, e, c in hits:
            if c in dup_lang:
                continue
            dup_lang.add(c)
            deduped.append((s, e, c))
        for i, (s, e, c) in enumerate(deduped):
            if c in seen:
                continue
            seen.add(c)
            next_lang_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(norm)
            in_seg = [(pos, lvl) for pos, lvl in cefr_matches if e <= pos < next_lang_start]
            if not in_seg:
                in_seg = [(pos, lvl) for pos, lvl in cefr_matches if s <= pos < next_lang_start]
            item: dict[str, str] = {"language": c, "source": "language_fallback:text_scan"}
            if in_seg:
                item["level"] = min(in_seg, key=lambda t: t[0])[1]
            elif cefr_matches:
                behind = [(pos, lvl) for pos, lvl in cefr_matches if pos < s]
                if behind:
                    item["level"] = max(behind, key=lambda t: t[0])[1]
            out.append(item)
            if len(out) >= max_languages:
                return out
    return out


def extract_open_vocabulary_skill_rows(
    text: str,
    catalog_skill_keys: set[str],
    min_confidence: float,
    max_rows: int = 80,
    rejected_out: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Emit skills read from skills/tools/languages (and related) sections, not limited to the DB catalog."""
    sections, section_weights = _extract_sections(text)
    section_phrases = _extract_section_phrases(sections)
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for phrase, section in section_phrases:
        if not _is_skill_heading(section):
            continue
        lang_sec = _is_language_heading(section)
        if lang_sec:
            # Languages are extracted in a dedicated channel.
            continue
        if not _open_vocab_phrase_ok(phrase, language_section=lang_sec):
            if rejected_out is not None:
                rejected_out.append(phrase.strip()[:120])
            continue
        display = _display_open_vocab_skill(phrase)
        if not display:
            continue
        sk = _skill_key(display)
        if not sk or sk in catalog_skill_keys or sk in seen_keys:
            continue
        seen_keys.add(sk)
        w = float(section_weights.get(section, 0.9))
        w = max(0.75, min(1.1, w))
        floor = max(float(min_confidence), 0.62)
        wc = len(display.split())
        if wc <= 4:
            conf = round(min(0.84, max(floor, 0.80 * w)), 2)
        else:
            conf = round(min(0.78, max(floor, 0.74 * w)), 2)
        if conf < min_confidence:
            continue
        sec_slug = re.sub(r"[^a-z0-9]+", "_", _normalize_text(section)[:48]).strip("_") or "section"
        rows.append(
            {
                "skill": display,
                "confidence": conf,
                "source": f"cv_section:{sec_slug}",
            }
        )
        if len(rows) >= max_rows:
            break
    rows.sort(key=lambda row: (-float(row["confidence"]), str(row["skill"])))
    return rows


def extract_soft_skill_rows(
    text: str,
    min_confidence: float = 0.6,
    max_rows: int = 20,
    fallback_mode: bool = False,
) -> list[dict[str, Any]]:
    """Extract soft/managerial competencies as a dedicated channel."""
    sections, _section_weights = _extract_sections(text)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    floor = max(float(min_confidence), 0.62 if fallback_mode else 0.66)
    for section, lines in sections.items():
        sec_key = _normalize_text(section)
        consider_section = (
            _is_skill_heading(section)
            or _is_experience_heading(section)
            or "competence" in sec_key
            or "compétence" in section.lower()
        )
        if not consider_section and not fallback_mode:
            continue
        for raw_line in lines:
            line = _normalize_text(raw_line)
            line_core = re.sub(r"[^a-z0-9\s]", " ", line)
            line_core = re.sub(r"\s+", " ", line_core).strip()
            if not line_core:
                continue
            for hint, canonical in _SOFT_SKILL_HINTS.items():
                if hint not in line_core:
                    continue
                sk = _skill_key(canonical)
                if not sk or sk in seen:
                    continue
                seen.add(sk)
                # vary confidence by section relevance and lexical match strength
                section_boost = 0.08 if (_is_experience_heading(section) or "projet" in sec_key or "project" in sec_key) else 0.04
                match_boost = min(0.06, max(0.0, (len(hint.split()) - 1) * 0.015))
                lexical_boost = (sum(ord(ch) for ch in canonical) % 4) * 0.01
                base = 0.68 if fallback_mode else 0.70
                conf = round(min(0.88, max(floor, base + section_boost + match_boost + lexical_boost)), 2)
                source = "softskill_fallback" if fallback_mode else f"softskill:{sec_key or 'section'}"
                out.append({"skill": canonical, "confidence": conf, "source": source})
                if len(out) >= max_rows:
                    return out
    out.sort(key=lambda row: (-float(row["confidence"]), str(row["skill"])))
    return out


def detect_skill_spans_with_ensemble(
    text: str,
    known_skills: Iterable[str],
    min_confidence: float = 0.6,
    semantic_threshold: float = 0.72,
    max_rows: int = 30,
    use_hf_ner: bool = False,
) -> list[dict[str, Any]]:
    """
    Lightweight NER/span-like matcher:
    - extracts short spans from skill/tool sections
    - semantically maps span -> nearest known skill
    - emits as a separate channel/source (`ner_span`)
    """
    known_list = [str(k).strip() for k in known_skills if isinstance(k, str) and str(k).strip()]
    if not known_list:
        return []
    skill_vectors, skill_names = _get_skill_embeddings(known_list)
    embedder = _get_embedder()
    if embedder is None or skill_vectors is None or not skill_names:
        return []

    sections, section_weights = _extract_sections(text)
    section_phrases = _extract_section_phrases(sections)
    span_items: list[tuple[str, str]] = []
    seen_span: set[str] = set()
    for phrase, section in section_phrases:
        if not _is_skill_heading(section) or _is_language_heading(section):
            continue
        if not _open_vocab_phrase_ok(phrase):
            continue
        norm = _normalize_text(phrase)
        if not norm or norm in seen_span:
            continue
        seen_span.add(norm)
        span_items.append((phrase, section))
        if len(span_items) >= 200:
            break
    if use_hf_ner:
        for phrase in _extract_hf_ner_spans(text):
            if not _open_vocab_phrase_ok(phrase):
                continue
            norm = _normalize_text(phrase)
            if not norm or norm in seen_span:
                continue
            seen_span.add(norm)
            span_items.append((phrase, "hf_ner"))
            if len(span_items) >= 260:
                break
    if not span_items:
        return []

    phrases = [p for p, _ in span_items]
    try:
        phrase_vectors = np.asarray(embedder.generate_embeddings(phrases), dtype=np.float32)
    except Exception:
        return []
    if phrase_vectors.size == 0:
        return []

    sims = phrase_vectors @ skill_vectors.T
    out: list[dict[str, Any]] = []
    seen_skill: set[str] = set()
    floor = max(float(min_confidence), 0.62)
    for i, (_phrase, section) in enumerate(span_items):
        idx = int(np.argmax(sims[i]))
        sim = float(sims[i, idx])
        if sim < semantic_threshold:
            continue
        skill = skill_names[idx]
        sk = _skill_key(skill)
        if not sk or sk in seen_skill:
            continue
        seen_skill.add(sk)
        weight = max(0.75, min(1.08, float(section_weights.get(section, 0.9))))
        raw = min(0.90, sim * weight)
        conf = _calibrate_confidence(raw, "ner_span", weight)
        if conf < floor:
            continue
        out.append({"skill": skill, "confidence": round(conf, 2), "source": f"ner_span:{section}"})
        if len(out) >= max_rows:
            break
    out.sort(key=lambda row: (-float(row["confidence"]), str(row["skill"])))
    return out


def _source_channel_family(source: str) -> str:
    """Coarse channel bucket used for multi-source confidence signals."""
    s = (source or "").strip().lower()
    if s.startswith("exact:") or s.startswith("fuzzy:") or s == "synonym":
        return "catalog"
    if s.startswith("cv_section:"):
        return "open_vocab"
    if s.startswith("ner_span"):
        return "ner"
    if s.startswith("semantic:"):
        return "semantic"
    if s.startswith("semantic_augment"):
        return "semantic_augment"
    if s.startswith("sentence_bullet"):
        return "sentence_bullet"
    if s.startswith("sentence_text"):
        return "sentence_text"
    if s.startswith("softskill"):
        return "softskill"
    if s.startswith("legacy"):
        return "legacy"
    return "other"


def _merge_skill_rows(
    catalog_rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in catalog_rows + extra_rows:
        if not isinstance(r, dict) or not r.get("skill"):
            continue
        k = _skill_key(str(r["skill"]))
        if not k:
            continue
        grouped.setdefault(k, []).append(dict(r))

    fused: list[dict[str, Any]] = []
    for _k, rows in grouped.items():
        if not rows:
            continue

        def _score(row: dict[str, Any]) -> float:
            try:
                return float(row.get("confidence", 0))
            except Exception:
                return 0.0

        best_row = max(rows, key=_score)
        best_score = _score(best_row)

        channels: set[str] = set()
        evidence: list[str] = []
        for row in rows:
            channels.add(_source_channel_family(str(row.get("source", ""))))
            for ev in row.get("evidence") or []:
                if not ev:
                    continue
                frag = _trim_evidence_fragment(str(ev), max_len=120)
                if frag and frag not in evidence:
                    evidence.append(frag)

        # Consolidate confidence from mention multiplicity + channel diversity.
        mention_bonus = min(0.08, max(0, len(rows) - 1) * 0.02)
        channel_bonus = min(0.06, max(0, len(channels) - 1) * 0.02)
        consolidated = min(0.98, max(0.01, best_score + mention_bonus + channel_bonus))

        out = dict(best_row)
        out["confidence"] = round(consolidated, 2)
        out["_conf_channels"] = channels or {_source_channel_family(str(best_row.get("source", "")))}
        if evidence:
            out["evidence"] = evidence[:4]
        fused.append(out)

    return sorted(fused, key=lambda row: (-float(row.get("confidence", 0)), str(row.get("skill", ""))))


def _skill_line_hit(skill: str, line: str) -> bool:
    patt = re.escape(_normalize_for_pattern(skill))
    if not patt:
        return False
    norm = _normalize_for_pattern(line)
    try:
        if re.search(rf"\b{patt}\b", norm):
            return True
    except re.error:
        return False
    for rx, can in _SKILL_SENTENCE_PATTERNS.items():
        if canonicalize_skill(can) != canonicalize_skill(skill):
            continue
        try:
            if re.search(rx, norm):
                return True
        except re.error:
            continue
    return False


def _sections_with_skill(skill: str, sections: dict[str, list[str]]) -> int:
    n = 0
    for _sec, lines in sections.items():
        for line in lines:
            if _skill_line_hit(skill, line):
                n += 1
                break
    return n


def _skill_in_bullet_lines(skill: str, text: str) -> bool:
    for raw in (text or "").splitlines():
        if not _is_bullet_line(raw):
            continue
        if _skill_line_hit(skill, raw):
            return True
    return False


def _skill_in_skill_section_lines(skill: str, sections: dict[str, list[str]]) -> bool:
    for sec, lines in sections.items():
        if not _is_skill_heading(sec):
            continue
        for line in lines:
            if _skill_line_hit(skill, line):
                return True
    return False


def _lines_in_doc_with_skill(skill: str, text: str) -> int:
    patt = re.escape(_normalize_for_pattern(skill))
    if not patt:
        return 0
    n = 0
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        norm = _normalize_for_pattern(raw)
        try:
            if re.search(rf"\b{patt}\b", norm):
                n += 1
                continue
        except re.error:
            pass
        for rx, can in _SKILL_SENTENCE_PATTERNS.items():
            if canonicalize_skill(can) != canonicalize_skill(skill):
                continue
            try:
                if re.search(rx, norm):
                    n += 1
                    break
            except re.error:
                continue
    return n


def enrich_skill_confidence_rows(
    rows: list[dict[str, Any]],
    text: str,
    sections: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Calibrate confidence: weak-signal penalties, cross-section rewards, rarity."""
    if not rows:
        return []
    cal_a, cal_b, cal_on = load_platt_params()
    sections = sections or {}
    nonempty_lines = max(1, sum(1 for ln in (text or "").splitlines() if ln.strip()))

    mention_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("skill"):
            continue
        sk = str(row["skill"]).strip()
        mention_counts[sk] = _lines_in_doc_with_skill(sk, text)
    max_mentions = max(mention_counts.values()) if mention_counts else 0

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("skill"):
            continue
        raw = dict(row)
        skill = str(raw["skill"]).strip()
        source = str(raw.get("source", ""))
        channels: set[str] = set(raw.pop("_conf_channels", None) or [])
        if not channels:
            channels = {_source_channel_family(source)}

        try:
            base = float(raw.get("confidence", 0))
        except Exception:
            base = 0.0

        patt = re.escape(_normalize_for_pattern(skill))
        mentions = mention_counts.get(skill, 0)
        freq_boost = min(0.12, max(0, mentions - 1) * 0.03)

        source_boost = 0.0
        if source.startswith("exact"):
            source_boost = 0.06
        elif source.startswith("fuzzy") or source == "synonym":
            source_boost = 0.04
        elif source.startswith("ner_span"):
            source_boost = 0.03
        elif source.startswith("sentence_bullet"):
            source_boost = 0.04
        elif source.startswith("sentence_text"):
            source_boost = 0.02
        elif source.startswith("softskill"):
            source_boost = 0.01
        elif source.startswith("semantic_augment"):
            source_boost = 0.02

        weak_penalty = 0.0
        if mentions <= 1:
            weak_penalty += 0.05
        if len(channels) <= 1:
            weak_penalty += 0.04

        section_hits = _sections_with_skill(skill, sections) if sections else 0
        cross_section_boost = 0.08 if section_hits >= 2 else 0.0

        dual_context_boost = 0.0
        if sections and _skill_in_bullet_lines(skill, text) and _skill_in_skill_section_lines(skill, sections):
            dual_context_boost = 0.06

        line_hits = mention_counts.get(skill, 0)
        line_frac = line_hits / nonempty_lines

        rarity_penalty = 0.0
        if line_frac > 0.35:
            rarity_penalty += min(0.06, (line_frac - 0.35) * 0.22)
        if max_mentions > 0 and mentions >= max_mentions and max_mentions >= 10:
            rarity_penalty += 0.04

        score = base + freq_boost + source_boost + cross_section_boost + dual_context_boost
        score -= weak_penalty + rarity_penalty
        score = max(0.52, min(0.94, score))
        if cal_on:
            score = apply_platt_on_unit_interval(score, cal_a, cal_b)
        upd = raw
        upd["confidence"] = round(max(0.01, min(0.99, score)), 2)
        out.append(upd)
    out.sort(key=lambda r: (-float(r.get("confidence", 0)), str(r.get("skill", ""))))
    return out


def _skill_category_from_source(source: str) -> str | None:
    src = (source or "").strip().lower()
    if src.startswith("softskill"):
        return "soft-skills"
    return None


def group_skills(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "management": [],
        "business": [],
        "technical": [],
        "soft-skills": [],
        "other": [],
    }
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        s = canonicalize_skill(str(row.get("skill", "")))
        if not s or s in seen:
            continue
        seen.add(s)
        cat = _SKILL_CATEGORY_MAP.get(s) or _skill_category_from_source(str(row.get("source", ""))) or "other"
        grouped.setdefault(cat, []).append(s)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped


def build_skill_hierarchy(skills: list[str]) -> list[dict[str, Any]]:
    skill_set = {canonicalize_skill(s or "") for s in skills if s}
    skill_set.discard("")
    out: list[dict[str, Any]] = []
    for parent, subs in _SKILL_HIERARCHY_MAP.items():
        present_subs = sorted([s for s in subs if s in skill_set])
        if parent in skill_set and present_subs:
            out.append({"parent": parent, "subskills": present_subs})
        elif parent not in skill_set and len(present_subs) >= 2:
            out.append({"parent": parent, "subskills": present_subs})
    out.sort(key=lambda x: x["parent"])
    return out


def _skill_mention_count(skill: str, text: str) -> int:
    patt = re.escape(_normalize_for_pattern(skill or ""))
    if not patt:
        return 0
    normalized_text = _normalize_for_pattern(text or "")
    try:
        return len(re.findall(rf"\b{patt}\b", normalized_text))
    except Exception:
        return 0


def build_skill_graph(skills: list[str], text: str) -> dict[str, dict[str, Any]]:
    skill_set = {canonicalize_skill(s or "") for s in skills if s}
    skill_set.discard("")
    graph: dict[str, dict[str, Any]] = {}
    for parent, subskills in _SKILL_HIERARCHY_MAP.items():
        detected_children = sorted([s for s in subskills if s in skill_set])
        if parent in skill_set or len(detected_children) >= 2:
            ratio = (len(detected_children) / max(1, len(subskills)))
            mention_boost = min(0.15, _skill_mention_count(parent, text) * 0.03)
            confidence = max(0.65, min(0.95, 0.65 + (0.25 * ratio) + mention_boost))
            graph[parent] = {
                "children": detected_children,
                "confidence": round(confidence, 2),
            }
    return graph


def _trim_evidence_fragment(frag: str, max_len: int = 88) -> str:
    s = re.sub(r"\s+", " ", (frag or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _expand_snippet_with_verb_prefix(norm_segment: str, m: re.Match[str]) -> str:
    frag = (m.group(0) or "").strip()
    if len(frag) >= 14:
        return frag
    start = m.start()
    prefix_tokens = norm_segment[:start].split()
    if len(prefix_tokens) >= 2:
        head = " ".join(prefix_tokens[-2:])
        return f"{head} {frag}".strip()
    if prefix_tokens:
        return f"{prefix_tokens[-1]} {frag}".strip()
    return frag


def _snippet_from_pattern_on_norm(norm_segment: str, skill: str) -> str | None:
    best: str | None = None
    for patt, can in _SKILL_SENTENCE_PATTERNS.items():
        if canonicalize_skill(can) != canonicalize_skill(skill):
            continue
        try:
            for m in re.finditer(patt, norm_segment):
                raw_frag = _expand_snippet_with_verb_prefix(norm_segment, m)
                frag = _trim_evidence_fragment(raw_frag)
                if best is None or len(frag) > len(best):
                    best = frag
        except re.error:
            continue
    return best


def _snippet_around_tokens(norm_segment: str, skill: str) -> str | None:
    phrase = _normalize_for_pattern(skill).strip()
    if len(phrase) < 2:
        return None
    words = norm_segment.split()
    pw = phrase.split()
    if pw and words:
        for i in range(0, len(words) - len(pw) + 1):
            if words[i : i + len(pw)] == pw:
                lo = max(0, i - 2)
                hi = min(len(words), i + len(pw) + 4)
                return _trim_evidence_fragment(" ".join(words[lo:hi]))
    if phrase in norm_segment:
        idx = norm_segment.find(phrase)
        chunk = norm_segment[max(0, idx - 18) : idx + len(phrase) + 32]
        return _trim_evidence_fragment(chunk)
    return None


def _extract_evidence_chunk(skill: str, line: str) -> str | None:
    raw = line.strip()
    for prefix in BULLET_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    segments = [s.strip() for s in re.split(r"\s*•\s*|\s*·\s*|\s*\|\s*", raw) if s.strip()]
    if not segments:
        segments = [raw]
    best: str | None = None
    for seg in segments:
        norm = _normalize_for_pattern(seg)
        if not norm:
            continue
        cand = _snippet_from_pattern_on_norm(norm, skill) or _snippet_around_tokens(norm, skill)
        if cand and (best is None or len(cand) < len(best)):
            best = cand
    return best


def _collect_skill_evidence(skill: str, text: str, max_items: int = 2) -> list[str]:
    if not skill or not (text or "").strip():
        return []
    evidence: list[str] = []
    lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
    for line in lines:
        if not _skill_line_hit(skill, line):
            continue
        chunk = _extract_evidence_chunk(skill, line)
        if chunk and chunk not in evidence:
            evidence.append(chunk)
        if len(evidence) >= max_items:
            break
    return evidence


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
    use_hf_ner: bool = False,
    use_semantic_augment: bool = False,
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
        "skills_grouped": {
            "management": [],
            "business": [],
            "technical": [],
            "soft-skills": [],
            "other": [],
        },
        "skill_hierarchy": [],
        "skill_graph": {},
        "extracted_languages": [],
        "language_details": [],
        "extraction_channels": {
            "catalog_match": [],
            "open_vocab": [],
            "soft_skill": [],
            "sentence": [],
            "semantic_augment": [],
            "language": [],
            "project_text": [],
        },
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

    layout_degraded, layout_warnings = _layout_quality_assessment(
        text=text,
        filename=safe_filename,
        file_bytes_len=len(safe_bytes),
    )
    if layout_degraded:
        result["degraded"] = True
    for w in layout_warnings:
        if w not in result["warnings"]:
            result["warnings"].append(w)

    if not text.strip():
        result["degraded"] = True
        result["warnings"].append("empty_text")
        return result

    section_map, _section_weights_header = _extract_sections(text)

    try:
        lang_details = extract_language_details_from_sections(text)
        fallback_details = extract_language_details_from_anywhere(
            text=text,
            existing_languages={str(item.get("language", "")) for item in lang_details if isinstance(item, dict)},
        )
        lang_details.extend(fallback_details)
        result["language_details"] = lang_details
        result["extracted_languages"] = [row.get("language", "") for row in lang_details if row.get("language")]
        result["extraction_channels"]["language"] = list(result["extracted_languages"])
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("extract_languages_from_sections", exc))

    try:
        skill_budget = DEFAULT_SKILL_TIME_BUDGET_SECONDS
        if len(text) > 80_000:
            skill_budget = 0.20
        start = time.perf_counter()
        rows: list[dict[str, Any]] = []
        if skills_list:
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

        catalog_keys = {
            _skill_key(str(r.get("skill", "")))
            for r in rows
            if isinstance(r, dict) and r.get("skill")
        }
        rejected_project_phrases: list[str] = []
        open_rows = extract_open_vocabulary_skill_rows(
            text=text,
            catalog_skill_keys=catalog_keys,
            min_confidence=safe_min_conf,
            rejected_out=rejected_project_phrases,
        )
        ner_rows = detect_skill_spans_with_ensemble(
            text=text,
            known_skills=skills_list,
            min_confidence=safe_min_conf,
            use_hf_ner=use_hf_ner,
        )
        soft_rows = extract_soft_skill_rows(text=text, min_confidence=safe_min_conf, fallback_mode=False)
        sentence_rows = extract_sentence_skill_rows(text=text, min_confidence=safe_min_conf, debug=False)
        rows = _merge_skill_rows(rows, open_rows + ner_rows + soft_rows + sentence_rows)
        soft_channel_names = [
            str(r.get("skill")).strip()
            for r in soft_rows
            if isinstance(r, dict) and r.get("skill")
        ]
        if not rows:
            fallback_soft_rows = extract_soft_skill_rows(
                text=text,
                min_confidence=safe_min_conf,
                fallback_mode=True,
            )
            rows = _merge_skill_rows(rows, fallback_soft_rows)
            soft_channel_names = [
                str(r.get("skill")).strip()
                for r in fallback_soft_rows
                if isinstance(r, dict) and r.get("skill")
            ]
        if use_semantic_augment and skills_list:
            exist_keys = {
                _skill_key(str(r.get("skill", "")))
                for r in rows
                if isinstance(r, dict) and r.get("skill")
            }
            aug_budget = float(skill_budget) if skill_budget and skill_budget > 0 else 0.45
            aug_budget = min(aug_budget, 0.48)
            augment_extra = augment_skills_semantically_gated(
                text=text,
                known_skills=skills_list,
                existing_skill_keys=exist_keys,
                min_confidence=safe_min_conf,
                time_budget_seconds=aug_budget,
            )
            rows = rows + augment_extra
        rows, negated_filtered = filter_negated_skill_rows(rows, text=text, window_lines=1)
        if negated_filtered:
            result["warnings"].append("negated_skills_filtered")
        rows = enrich_skill_confidence_rows(rows, text=text, sections=section_map)
        rows_with_evidence: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("skill"):
                continue
            enhanced = dict(row)
            prior_ev = list(enhanced.get("evidence") or [])
            collected = _collect_skill_evidence(str(row.get("skill", "")), text=text)
            enhanced["evidence"] = list(dict.fromkeys(prior_ev + collected))[:3]
            rows_with_evidence.append(enhanced)
        rows = rows_with_evidence
        apply_weak_hedge_penalty_to_rows(rows)
        attach_confidence_normalized(rows)

        result["extracted_skills"] = rows
        result["skills"] = [
            str(r.get("skill")).strip()
            for r in rows
            if isinstance(r, dict) and r.get("skill")
        ]
        result["skills_grouped"] = group_skills(rows)
        result["skill_hierarchy"] = build_skill_hierarchy(result["skills"])
        result["skill_graph"] = build_skill_graph(result["skills"], text=text)
        catalog_names = [
            str(r.get("skill")).strip()
            for r in rows
            if isinstance(r, dict)
            and r.get("skill")
            and not str(r.get("source", "")).startswith("cv_section:")
            and not str(r.get("source", "")).startswith("ner_span:")
            and not str(r.get("source", "")).startswith("softskill")
            and not str(r.get("source", "")).startswith("semantic_augment")
        ]
        open_vocab_names = [
            str(r.get("skill")).strip()
            for r in rows
            if isinstance(r, dict)
            and r.get("skill")
            and str(r.get("source", "")).startswith("cv_section:")
        ]
        soft_skill_names = sorted({s for s in soft_channel_names if s}, key=lambda x: x.lower())
        sentence_names = [
            str(r.get("skill")).strip()
            for r in rows
            if isinstance(r, dict)
            and r.get("skill")
            and str(r.get("source", "")).startswith("sentence_")
        ]
        semantic_augment_names = [
            str(r.get("skill")).strip()
            for r in rows
            if isinstance(r, dict)
            and r.get("skill")
            and str(r.get("source", "")).startswith("semantic_augment")
        ]
        result["extraction_channels"]["catalog_match"] = catalog_names
        result["extraction_channels"]["open_vocab"] = open_vocab_names
        result["extraction_channels"]["soft_skill"] = soft_skill_names
        result["extraction_channels"]["sentence"] = sentence_names
        result["extraction_channels"]["semantic_augment"] = semantic_augment_names
        result["extraction_channels"]["project_text"] = sorted(
            {p for p in rejected_project_phrases if p},
            key=lambda x: x.lower(),
        )[:80]
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
            legacy_rows = [
                {"skill": str(s).strip(), "confidence": 0.6, "source": "legacy", "evidence": []}
                for s in legacy
                if isinstance(s, str) and str(s).strip()
            ]
            legacy_rows, negated_filtered = filter_negated_skill_rows(legacy_rows, text=text, window_lines=1)
            result["skills"] = [str(r.get("skill")).strip() for r in legacy_rows if r.get("skill")]
            if negated_filtered:
                result["warnings"].append("negated_skills_filtered")
            apply_weak_hedge_penalty_to_rows(legacy_rows)
            attach_confidence_normalized(legacy_rows)
            result["extracted_skills"] = legacy_rows
            result["skills_grouped"] = group_skills(result["extracted_skills"])
            result["skill_hierarchy"] = build_skill_hierarchy(result["skills"])
            result["skill_graph"] = build_skill_graph(result["skills"], text=text)
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
