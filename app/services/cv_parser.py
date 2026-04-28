import io
import json
import re
import math
import time
import os
import unicodedata
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable
import numpy as np
import pdfplumber
from docx import Document
from app.ai.confidence_calibration import apply_platt_on_unit_interval, load_platt_params
from app.ai.skill_canonicalization import canonicalize_skill
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_stage_error = lambda stage, exc: f"{stage}:{type(exc).__name__}:{str(exc)}"

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
_CERTIFICATION_HEADING_HINTS = (
    "certification",
    "certifications",
    "certificat",
    "certificats",
    "certificate",
    "certificates",
    "license",
    "licenses",
    "diploma",
    "diplomas",
    "training",
    "formations",
    "cours",
    "diplomes",
    "attestations",
    "certifs",
    "formation en",
    "certificats obtenus",
)
_PROJECT_HEADING_HINTS = (
    "project",
    "projects",
    "personal projects",
    "academic projects",
    "professional projects",
    "hands on projects",
    "hands-on projects",
    "hands on security projects",
    "hands-on security projects",
    "security projects",
    "practical projects",
    "hands on",
    "hands-on",
    "portfolio",
    "realisation",
    "réalisation",
    "projets",
    "projets realises",
    "projets pratiques",
    "mes projets",
    "travaux",
    "realisations",
    "applications developpees",
)
_LANGUAGE_HEADING_HINTS = ("language", "languages", "langue", "langues", "idioma", "idiomas")
_DURATION_PHRASE_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:months?|years?|yrs?)$", re.I)
WORD_RE = re.compile(r"[a-z0-9+.#/\-]+")
_EXPERIENCE_YEARS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|ans)\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*(present|current|now|19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_LEADING_DATE_RANGE_RE = re.compile(
    r"^\s*\d{1,2}\s+[A-Za-z]+\s*[-–—]\s*\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?\s+",
    re.IGNORECASE,
)
_DATE_RANGE_RE = re.compile(r"^(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}$")
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
    "formation",
    "formations",
    "diploma",
    "diplomas",
    "diplome",
    "diplomes",
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
    "etudiante",
    "etudiant",
    "stagiaire",
    "alternant",
    "apprenti",
    "ingenieure",
    "ingenieur",
    "developpeur",
    "developpeuse",
    "analyste",
    "concepteur",
    "conceptrice",
)
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().\-]{6,}\d)(?!\w)")
_OBFUSCATED_AT_RE = re.compile(
    r"(?i)(?<=[A-Z0-9._%+\-])\s*(?:\[\s*at\s*\]|\(\s*at\s*\)|\bat\b)\s*(?=[A-Z0-9.\-])"
)
_OBFUSCATED_DOT_RE = re.compile(r"(?i)(?<=[A-Z0-9])\s*(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\bdot\b)\s*(?=[A-Z0-9])")
_NAME_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'`-]{0,29}$")
_CONTACT_URL_HINTS = ("http://", "https://", "www.", "linkedin", "github")
_CONTACT_LABEL_HINTS = ("phone", "telephone", "tel", "mobile", "gsm", "email", "mail")
_CONTACT_SCAN_CHARS = 6000
_PREVIEW_CONTACT_CHARS = 1200
_NAME_STOPWORDS = frozenset(
    {
        "curriculum",
        "vitae",
        "resume",
        "cv",
        "candidate",
        "profile",
        "linkedin",
        "github",
        "tunisia",
        "tunisie",
        "tunis",
        "france",
        "paris",
        "junior",
        "senior",
        "lead",
        "principal",
        "security",
        "penetration",
        "tester",
        "engineer",
        "developer",
        "analyst",
        "manager",
        "student",
        "intern",
    }
)
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
    "hands on projects",
    "hands-on projects",
    "hands on security projects",
    "hands-on security projects",
    "security projects",
    "practical projects",
    "career objective",
    "education and training",
    "additional information",
    "competences techniques",
    "compétences techniques",
    "experience professionnelle",
    "expérience professionnelle",
    "competences particulieres",
    "compétences particulières",
    "activite extra professionnelle",
    "activite extra-professionnelle",
    "activites extra professionnelles",
    "activites extra-professionnelles",
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


def _env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError:
            value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return float(value)


def _is_skill_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _SKILL_HEADING_HINTS)


def _is_language_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _LANGUAGE_HEADING_HINTS)

def _is_certification_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _CERTIFICATION_HEADING_HINTS)


def _is_project_heading(section_key: str) -> bool:
    key = _normalize_text(section_key)
    return any(h in key for h in _PROJECT_HEADING_HINTS)


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
    normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
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


def _looks_like_semantic_section_heading(key: str, words: list[str]) -> bool:
    """Semantic fallback for common CV headings that are not ALL-CAPS."""
    if not key or not words:
        return False
    if len(words) > 8:
        return False
    if re.search(r"[,;|]", key):
        return False
    if _is_certification_heading(key) or _is_language_heading(key):
        return True

    if _is_project_heading(key):
        if key in {"project", "projects"}:
            return True
        if key.endswith(" project") or key.endswith(" projects"):
            return True
        if "hands on" in key or "hands-on" in key or "portfolio" in key:
            return True

    if _is_skill_heading(key):
        if key in _SINGLE_WORD_RESUME_HEADINGS:
            return True
        if key.endswith(" skills") or key.endswith(" tools") or key.endswith(" technologies"):
            return True
        if key.endswith(" frameworks") or key.endswith(" stack"):
            return True

    if _is_experience_heading(key) or _is_education_heading(key):
        return key in {
            "experience",
            "work experience",
            "professional experience",
            "employment",
            "education",
            "education and training",
            "formation",
            "formations",
        }
    return False


def _matches_resume_section_heading(key: str, words: list[str]) -> bool:
    """True if this short line is a typical CV section title, not a skill/tool bullet."""
    if not key or not words:
        return False
    if key in _SINGLE_WORD_RESUME_HEADINGS:
        return True
    for phrase in _MULTI_WORD_RESUME_HEADINGS:
        if key == phrase or key.startswith(phrase + " ") or key.endswith(" " + phrase):
            return True
    if len(words) <= 4:
        first = _normalize_text(words[0])
        if first in (
            "competence",
            "competences",
            "langue",
            "langues",
            "formation",
            "formations",
            "activite",
            "activites",
            "certification",
            "certifications",
        ):
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
    if _looks_like_semantic_section_heading(key, words):
        return True
    return False


def _build_skill_index(known_skills: Iterable[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    catalog_noise_hints = (
        "lycee",
        "lycée",
        "universite",
        "université",
        "baccalaureat",
        "baccalauréat",
        "ecole",
        "école",
        "high school",
        "school",
        "college",
        "campus",
    )
    for raw in known_skills or []:
        canonical = canonicalize_skill(raw)
        if not canonical:
            continue
        canonical_low = _normalize_text(canonical)
        if any(h in canonical_low for h in catalog_noise_hints) and len(canonical_low.split()) >= 2:
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
    if any(line.startswith(prefix) for prefix in BULLET_PREFIXES):
        return False, "", 0.0

    for prefix in BULLET_PREFIXES:
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
            break

    trimmed = line.strip(":").strip()
    if not trimmed or len(trimmed) > 80:
        return False, "", 0.0

    if re.search(r"[,;|]", trimmed):
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

    short_all_caps_not_section = (
        not has_colon
        and not section_like
        and len(words) == 1
        and len(trimmed) <= 5
        and trimmed.isalpha()
        and trimmed.isupper()
    )

    short_upper_phrase_not_section = (
        not has_colon
        and not section_like
        and upper_ratio >= 0.85
        and 1 <= len(words) <= 3
        and len(trimmed) <= 24
    )

    is_heading = has_colon or section_like or (
        upper_ratio >= 0.6
        and not short_all_caps_not_section
        and not short_upper_phrase_not_section
    )
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


_SPAN_NEGATION_RE = re.compile(
    r"(?i)"
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\blacking\b|\black\s+of\b|\bneither\b|\bnor\b|"
    r"\bno\s+experience\b|\bwithout\s+experience\b|"
    r"\bsans\b|\baucune\b|\baucun\b|\bjamais\b|"
    r"\bpas\s+de\b|\bpas\s+d['\u2019]|"
    r"\bni\s+l['\u2019]?\s*experience\b|\bzero\s+experience\b|\bz[eé]ro\s+exp[eé]rience\b)"
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
_CONTEXT_POSITIVE_RE = re.compile(
    r"(?i)\b("
    r"built|build|implemented|implementation|developed|development|designed|design|"
    r"deployed|deploy|created|create|delivered|delivery|maintained|maintain|"
    r"optimized|optimize|led|lead|shipped|architected|automated|automation|"
    r"contributed|contribution|owned|integrated|integration|"
    r"concu|concevoir|developpe|developpement|développé|développement|"
    r"mis\s+en\s+place|realise|réalisé"
    r")\b"
)
_CONTEXT_NEGATIVE_RE = re.compile(
    r"(?i)\b("
    r"interested\s+in|interest\s+in|looking\s+to\s+learn|want\s+to\s+learn|"
    r"learning|learned|beginner|novice|entry\s*level|"
    r"familiar\s+with|familiarity\s+with|basic|basics|limited|intro(?:duction)?\s+to|"
    r"awareness\s+of|exposure\s+to|notions?\s+of|"
    r"interesse\s+par|interesse\s+a|en\s+cours\s+d['\u2019]apprentissage|apprentissage|debutant|débutant|"
    r"connaissances?\s+de\s+base|notions?\s+de"
    r")\b"
)


def _span_text_negated(span: str) -> bool:
    if not (span or "").strip():
        return False
    norm = _normalize_for_pattern(span)
    return bool(_SPAN_NEGATION_RE.search(norm))


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


def _context_strength_from_evidence(evidence: list[str]) -> float:
    if not evidence:
        return 0.5
    positives = 0
    negatives = 0
    for ev in evidence:
        norm = _normalize_for_pattern(str(ev or ""))
        if not norm:
            continue
        if _span_text_negated(norm):
            negatives += 2
        if _CONTEXT_NEGATIVE_RE.search(norm) or _evidence_weak_hedge(norm):
            negatives += 1
        if _CONTEXT_POSITIVE_RE.search(norm):
            positives += 1
    total = max(1, positives + negatives)
    raw = 0.5 + (0.35 * (positives / total)) - (0.45 * (negatives / total))
    return float(max(0.0, min(1.0, raw)))


def apply_context_strength_to_rows(rows: list[dict[str, Any]]) -> None:
    """Attach context_strength and nudge confidence using local evidence quality."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        evidence = [str(item) for item in (row.get("evidence") or []) if str(item or "").strip()]
        strength = _context_strength_from_evidence(evidence)
        row["context_strength"] = round(strength, 4)
        try:
            base = float(row.get("confidence", 0.0))
        except Exception:
            base = 0.0
        delta = (strength - 0.5) * 0.24
        adjusted = max(0.52, min(0.99, base + delta))
        row["confidence"] = round(adjusted, 2)


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


def _source_conf_threshold(source: str, min_confidence: float) -> float:
    """Post-merge confidence floor per extraction source."""
    s = (source or "").strip().lower()
    base = max(0.0, min(1.0, float(min_confidence)))
    if s.startswith("sentence_text"):
        return max(0.50, base - 0.10)
    if s.startswith("sentence_bullet"):
        return max(0.50, base - 0.08)
    if s.startswith("softskill"):
        return max(0.50, base - 0.08)
    return -1.0


def apply_post_merge_source_gate(rows: list[dict[str, Any]], min_confidence: float) -> tuple[list[dict[str, Any]], int]:
    """Filter rows with source-aware confidence thresholds."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if not isinstance(row, dict) or not row.get("skill"):
            continue
        try:
            conf = float(row.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        try:
            conf_norm = float(row.get("confidence_normalized", 0.0))
        except Exception:
            conf_norm = 0.0
        src = str(row.get("source", ""))
        floor = _source_conf_threshold(src, min_confidence=min_confidence)
        if floor >= 0.0 and (conf + 1e-9 < floor) and (conf_norm < 0.58):
            dropped += 1
            continue
        kept.append(row)
    kept.sort(key=lambda r: (-float(r.get("confidence", 0.0)), str(r.get("skill", ""))))
    return kept, dropped


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(dict(out[key]), value)
        else:
            out[key] = value
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
        "exact": 1.05,  
        "synonym": 0.98,  
        "fuzzy": 0.90,
        "semantic": 0.85,  
        "semantic_augment": 0.84,
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
            if _is_education_heading(section):
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
_CERTIFICATION_MARKER_RE = re.compile(
    r"\b(?:certified|certificate|certificates|certification|certifications|certificat|certificats|diploma|diplomas|license|licensed)\b",
    re.IGNORECASE,
)
_CERTIFICATION_ACRONYM_RE = re.compile(
    r"\b(?:"
    r"ccna|ccnp|ccie|cissp|ceh|oscp|oswe|osce|ejpt|ecppt|pnpt|ewpt|"
    r"security\+|network\+|a\+|az-\d{3}|sc-\d{3}|dp-\d{3}|"
    r"gpen|gsec|gwapt|crt|crtp|crte|crto|crtl|pjpt|cpts|cbbh"
    r")\b",
    re.IGNORECASE,
)
_CERTIFICATION_PROVIDER_HINTS = frozenset(
    {
        "cisco",
        "comptia",
        "isc2",
        "offsec",
        "offensive security",
        "elearnsecurity",
        "ine security",
        "sans",
        "ec council",
        "microsoft",
        "aws",
        "google cloud",
        "hack the box",
        "pro labs",
        "tryhackme",
        "portswigger",
        "academy",
    }
)
_CERTIFICATION_ALIAS_HINTS = frozenset(
    {
        "ejpt",
        "ecppt",
        "ccna",
        "ccnp",
        "cissp",
        "ceh",
        "oscp",
        "oswe",
        "mythical",
        "puppet",
    }
)
_SKILL_SECTION_NOISE_TERMS = frozenset(
    {
        "skills",
        "technical skills",
        "core skills",
        "frameworks",
        "tools",
        "technologies",
        "hands on security projects",
        "hands-on security projects",
        "hands on projects",
        "hands-on projects",
        "security projects",
        "projects",
        "project",
        "certifications",
        "certification",
    }
)
_SKILL_LOCATION_NOISE_TERMS = frozenset(
    {
        "tunisie",
        "tunisia",
        "tunis",
        "sfax",
        "sousse",
        "france",
        "paris",
        "morocco",
        "algeria",
        "algerie",
    }
)
_SKILL_LEADING_NUMBER_NOISE_RE = re.compile(r"^\d+\s+[a-z].{2,}$")
_SKILL_CERT_SPLIT_NOISE_RE = re.compile(r"^\d+\s*-\s*[a-z].{2,}$")
_PROJECT_ACTION_HINT_RE = re.compile(
    r"\b(?:built|build|developed|developing|implemented|created|designed|deployed|led|worked on|contributed|réalisé|realise|realised|developpe|développé)\b",
    re.IGNORECASE,
)
_INLINE_LABEL_SPLIT_RE = re.compile(r"^\s*([^:]{2,64}?)\s*:\s*(.+?)\s*$")
_EVIDENCE_NOISE_LEAD_RE = re.compile(
    r"^(?:including|strong|advanced|expertise\s+in|experience\s+in)\s+",
    re.IGNORECASE,
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
    s = re.sub(r"^[+*#=~|:;,.!/?\\\-\s]+", "", s).strip()
    s = _LEADING_DATE_RANGE_RE.sub("", s).strip()
    s = re.sub(r"^\d{4}\s+", "", s).strip()
    s = re.sub(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\s+", "", s).strip()
    return s


def _is_noise_line(raw: str) -> bool:
    low = _normalize_text(raw)
    if not low:
        return True
    return any(h in low for h in _NOISE_LINE_HINTS)


def _normalize_for_pattern(text: str) -> str:
    low = _normalize_text(text)
    return unicodedata.normalize("NFKD", low).encode("ascii", "ignore").decode("ascii")


def _is_bullet_line(raw: str) -> bool:
    line = (raw or "").strip()
    return line.startswith(BULLET_PREFIXES)


def _split_inline_labeled_phrase(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if not text:
        return "", ""
    m = _INLINE_LABEL_SPLIT_RE.match(text)
    if not m:
        return "", text
    label = _normalize_text(m.group(1))
    tail = m.group(2).strip()
    return label, tail


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


def _normalize_board_entry(raw: str) -> str:
    value = _strip_open_vocab_leading_junk(str(raw or ""))
    value = _strip_parenthetical_qualifiers(value)
    value = re.sub(r"^\s*(?:[-*•]|â€¢)\s*", "", value).strip()
    value = re.sub(r"\s+", " ", value).strip(" .,:;|")
    return value[:220]


def _normalized_heading_from_line(raw_line: str) -> str:
    line = str(raw_line or "")[:MAX_LINE_CHARS].strip()
    if not line:
        return ""

    is_heading, key, _weight = _heading_candidate(line)
    if is_heading and key:
        return key

    candidate = line
    for prefix in BULLET_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            break
    if not candidate:
        return ""
    candidate = candidate.strip(":").strip()
    if not candidate:
        return ""
    words = candidate.split()
    key = _normalize_text(candidate)
    if _looks_like_semantic_section_heading(key, words):
        return key
    return ""


def _scan_board_lines_by_heading(text: str, *, target: str, max_lines: int = MAX_LINES) -> list[str]:
    active: str | None = None
    out: list[str] = []
    for raw in (text or "").splitlines()[:max_lines]:
        line = str(raw or "")[:MAX_LINE_CHARS].strip()
        if not line:
            continue
        heading_key = _normalized_heading_from_line(line)
        if heading_key:
            if _is_certification_heading(heading_key):
                active = "certification"
            elif _is_project_heading(heading_key):
                active = "project"
            else:
                active = None
            continue
        if active == target:
            out.append(line)
    return out


def _split_certification_candidates(candidate: str) -> list[str]:
    text = str(candidate or "").strip()
    if not text:
        return []
    if _CERTIFICATION_ACRONYM_RE.search(text) and re.search(r"\b\d\s*,\s*\d", text):
        return [text]
    return [part.strip() for part in re.split(r"[|;]+", text) if part and part.strip()]


def _looks_like_certification_entry(raw: str) -> bool:
    value = _normalize_board_entry(raw)
    if not value:
        return False
    low = _normalize_for_pattern(value)
    if not low:
        return False
    if low in _SKILL_SECTION_NOISE_TERMS:
        return False
    if _is_project_heading(low):
        return False
    if re.fullmatch(r"\d+", low):
        return False
    if _SKILL_CERT_SPLIT_NOISE_RE.match(low):
        return False
    if _SKILL_LEADING_NUMBER_NOISE_RE.match(low) and not _CERTIFICATION_ACRONYM_RE.search(low):
        return False
    if _PROJECT_ACTION_HINT_RE.search(low) and not _CERTIFICATION_MARKER_RE.search(low):
        return False

    if _CERTIFICATION_MARKER_RE.search(value):
        return True
    if _CERTIFICATION_ACRONYM_RE.search(low):
        return True
    if any(hint in low for hint in _CERTIFICATION_PROVIDER_HINTS):
        return True
    if any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in _CERTIFICATION_ALIAS_HINTS):
        return True

    if re.search(r"\(\s*(19|20)\d{2}\s*\)\s*$", raw) and len(raw.split()) >= 2:
        return True

    return False


def _looks_like_project_entry(raw: str) -> bool:
    value = _normalize_board_entry(raw)
    if not value:
        return False
    low = _normalize_for_pattern(value)
    if not low:
        return False
    if low in _SKILL_SECTION_NOISE_TERMS:
        return False
    if _looks_like_certification_entry(value):
        return False
    if _looks_like_skill_inventory_line(value):
        return False
    is_all_caps_acronym = raw.isupper() and len(raw) >= 3
    if len(low.split()) < 2 and not is_all_caps_acronym and not _PROJECT_ACTION_HINT_RE.search(low):
        return False
    return True


def _looks_like_project_detail_line(raw: str) -> bool:
    value = _normalize_board_entry(raw)
    if not value:
        return False
    low = _normalize_for_pattern(value)
    if not low:
        return False
    if _PROJECT_ACTION_HINT_RE.search(low):
        return False
    if _looks_like_certification_entry(value):
        return False
    if _is_project_heading(low) or _is_skill_heading(low):
        return False
    if re.fullmatch(r"\d+", low):
        return False
    parts = [part.strip() for part in re.split(r"[,\|;/]+", value) if part.strip()]
    if len(parts) >= 2 and len(low.split()) <= 20:
        return True
    return False


def _looks_like_skill_inventory_line(line: str) -> bool:
    normalized = _normalize_text(line)
    if not normalized:
        return True
    parts = [part.strip() for part in re.split(r"[,\|;/&]+", normalized) if part.strip()]
    if len(parts) < 3:
        return False
    short_parts = sum(1 for part in parts if len(part.split()) <= 3)
    return short_parts >= max(3, len(parts) - 1)


def _dedupe_board_entries(items: list[str], max_items: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = _normalize_board_entry(raw)
        if not value:
            continue
        key = _normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def extract_certifications_from_sections(text: str, max_items: int = 40) -> list[str]:
    sections, _weights = _extract_sections(text)
    entries: list[str] = []
    for section, lines in sections.items():
        if not _is_certification_heading(section):
            continue
        for raw_line in lines:
            line = raw_line[:MAX_LINE_CHARS].strip()
            if not line or _is_noise_line(line):
                continue
            heading_key = _normalized_heading_from_line(line)
            if heading_key:
                if _is_certification_heading(heading_key):
                    continue
                if (
                    _is_project_heading(heading_key)
                    or _is_skill_heading(heading_key)
                    or _is_experience_heading(heading_key)
                    or _is_education_heading(heading_key)
                    or _is_language_heading(heading_key)
                ):
                    break
            inline_label, inline_tail = _split_inline_labeled_phrase(line)
            candidate = inline_tail if inline_label and _is_certification_heading(inline_label) else line
            for part in _split_certification_candidates(candidate):
                if not _looks_like_certification_entry(part):
                    continue
                cleaned = _normalize_board_entry(part)
                if not cleaned:
                    continue
                entries.append(cleaned)

    if len(entries) < max_items:
        scanned_lines = _scan_board_lines_by_heading(text, target="certification")
        for line in scanned_lines:
            inline_label, inline_tail = _split_inline_labeled_phrase(line)
            candidate = inline_tail if inline_label and _is_certification_heading(inline_label) else line
            for part in _split_certification_candidates(candidate):
                if not _looks_like_certification_entry(part):
                    continue
                cleaned = _normalize_board_entry(part)
                if not cleaned:
                    continue
                entries.append(cleaned)
                if len(entries) >= max_items:
                    break
            if len(entries) >= max_items:
                break

    return _dedupe_board_entries(entries, max_items=max_items)


def extract_hands_on_projects_from_sections(text: str, max_items: int = 40) -> list[str]:
    sections, _weights = _extract_sections(text)
    projects: list[dict[str, Any]] = []

    def _push_project_title(raw: str) -> None:
        if len(projects) >= max_items:
            return
        title = _normalize_board_entry(raw)
        if not title:
            return
        if projects and _normalize_text(projects[-1].get("title", "")) == _normalize_text(title):
            return
        projects.append({"title": title, "details": []})

    def _push_project_detail(raw: str) -> None:
        if not projects:
            return
        detail = _normalize_board_entry(raw)
        if not detail:
            return
        details = projects[-1]["details"]
        key = _normalize_text(detail)
        seen = {_normalize_text(str(item or "")) for item in details}
        if key and key not in seen:
            details.append(detail)

    for section, lines in sections.items():
        section_is_project = _is_project_heading(section)
        if not section_is_project:
            continue
        for raw_line in lines:
            line = raw_line[:MAX_LINE_CHARS].strip()
            if not line or _is_noise_line(line):
                continue
            heading_key = _normalized_heading_from_line(line)
            if heading_key:
                if _is_project_heading(heading_key):
                    continue
                if (
                    _is_certification_heading(heading_key)
                    or _is_skill_heading(heading_key)
                    or _is_experience_heading(heading_key)
                    or _is_education_heading(heading_key)
                    or _is_language_heading(heading_key)
                ):
                    break
            inline_label, inline_tail = _split_inline_labeled_phrase(line)
            candidate = inline_tail if inline_label and _is_project_heading(inline_label) else line
            cleaned = _normalize_board_entry(candidate)
            if not cleaned:
                continue
            if _looks_like_project_detail_line(candidate) and projects:
                _push_project_detail(cleaned)
                continue
            if not _looks_like_project_entry(candidate):
                continue
            _push_project_title(cleaned)
            if len(projects) >= max_items:
                return [str(item.get("title", "")).strip() for item in projects if str(item.get("title", "")).strip()]

    if len(projects) < max_items:
        scanned_lines = _scan_board_lines_by_heading(text, target="project")
        for line in scanned_lines:
            inline_label, inline_tail = _split_inline_labeled_phrase(line)
            candidate = inline_tail if inline_label and _is_project_heading(inline_label) else line
            cleaned = _normalize_board_entry(candidate)
            if not cleaned:
                continue
            if _looks_like_project_detail_line(candidate) and projects:
                _push_project_detail(cleaned)
                continue
            if not _looks_like_project_entry(candidate):
                continue
            _push_project_title(cleaned)
            if len(projects) >= max_items:
                break

    if len(projects) < max_items:
        for raw_line in (text or "").splitlines()[:MAX_LINES]:
            line = raw_line[:MAX_LINE_CHARS].strip()
            if not line:
                continue
            inline_label, inline_tail = _split_inline_labeled_phrase(line)
            if inline_label and not _is_project_heading(inline_label):
                continue
            candidate = inline_tail if inline_tail else line
            cleaned = _normalize_board_entry(candidate)
            if not cleaned:
                continue
            if _looks_like_project_detail_line(cleaned) and projects:
                _push_project_detail(cleaned)
                continue
            if not _looks_like_project_entry(cleaned):
                continue
            if not _PROJECT_ACTION_HINT_RE.search(cleaned):
                continue
            _push_project_title(cleaned)
            if len(projects) >= max_items:
                break

    return [str(item.get("title", "")).strip() for item in projects if str(item.get("title", "")).strip()][:max_items]


def build_project_skill_links(
    *,
    projects: list[str],
    extracted_rows: list[dict[str, Any]],
    max_links: int = 120,
) -> list[dict[str, Any]]:
    if not projects or not extracted_rows:
        return []
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for project in projects:
        if len(links) >= max_links:
            break
        project_raw = re.sub(r"\s+", " ", str(project or "")).strip()
        if not project_raw:
            continue
        project_norm = _normalize_for_pattern(project_raw)
        if not project_norm:
            continue
        project_tokens = set(_tokenize(project_norm))
        for row in extracted_rows:
            if len(links) >= max_links:
                break
            if not isinstance(row, dict) or not row.get("skill"):
                continue
            skill = canonicalize_skill(str(row.get("skill", "")))
            if not skill:
                continue
            pair_key = (project_norm, _skill_key(skill))
            if pair_key in seen:
                continue
            if not _skill_line_hit(skill, project_norm):
                continue

            evidence = project_raw[:220]
            row_evidence = [str(item) for item in (row.get("evidence") or []) if str(item or "").strip()]
            if row_evidence:
                evidence = row_evidence[0][:220]
            try:
                row_conf = float(row.get("confidence", 0.0))
            except Exception:
                row_conf = 0.0
            try:
                context_strength = float(row.get("context_strength", 0.5))
            except Exception:
                context_strength = 0.5

            skill_tokens = {tok for tok in _tokenize(_normalize_for_pattern(skill)) if tok}
            overlap = 0.0
            if skill_tokens:
                overlap = len(skill_tokens & project_tokens) / max(1, len(skill_tokens))
            link_conf = max(0.0, min(1.0, (0.65 * row_conf) + (0.25 * context_strength) + (0.10 * overlap)))
            links.append(
                {
                    "project": project_raw,
                    "skill": skill,
                    "evidence_span": evidence,
                    "confidence": round(link_conf, 4),
                }
            )
            seen.add(pair_key)

    links.sort(key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("skill", ""))))
    return links[:max_links]


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
    if s.startswith("context_board:certification"):
        return "certification"
    if s.startswith("context_board:hands_on_project"):
        return "hands_on_project"
    if s.startswith("sentence_bullet"):
        return "sentence_bullet"
    if s.startswith("sentence_text"):
        return "sentence_text"
    if s.startswith("softskill"):
        return "softskill"
    if s.startswith("legacy"):
        return "legacy"
    return "other"


def _source_label_key(source: str) -> str:
    fam = _source_channel_family(source)
    if fam == "catalog":
        return "exact"
    if fam == "open_vocab":
        return "section"
    if fam == "ner":
        return "ner"
    if fam == "semantic":
        return "semantic"
    if fam == "semantic_augment":
        return "augment"
    if fam == "certification":
        return "section"
    if fam == "hands_on_project":
        return "section"
    if fam.startswith("sentence"):
        return "sentence"
    if fam == "softskill":
        return "softskill"
    if fam == "legacy":
        return "legacy"
    return "other"


def _confidence_band_for_source(source: str, confidence: float) -> str:
    fam = _source_channel_family(source)
    c = max(0.0, min(1.0, float(confidence)))
    if fam in {"catalog", "semantic", "ner"}:
        if c >= 0.78:
            return "high"
        if c >= 0.64:
            return "medium"
        return "low"
    if fam in {"open_vocab", "sentence_bullet", "sentence_text", "semantic_augment", "certification", "hands_on_project"}:
        if c >= 0.70:
            return "medium"
        return "low"
    if fam in {"softskill", "legacy"}:
        if c >= 0.72:
            return "medium"
        return "low"
    return "low"


def _merge_skill_rows(
    catalog_rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in catalog_rows + extra_rows:
        if not isinstance(r, dict) or not r.get("skill"):
            continue
        k = _skill_key(str(r["skill"]))
        if not k:
            continue
        kind = _source_channel_family(str(r.get("source", "")))
        prev = best.get(k)
        try:
            sc = float(r.get("confidence", 0))
        except Exception:
            sc = 0.0
        if prev is None:
            nr = dict(r)
            nr["_conf_channels"] = {kind}
            best[k] = nr
            continue
        try:
            ps = float(prev.get("confidence", 0))
        except Exception:
            ps = 0.0
        pchannels = set(prev.get("_conf_channels") or [])
        if not pchannels:
            pchannels = {_source_channel_family(str(prev.get("source", "")))}
        pchannels.add(kind)
        if sc > ps:
            nr = dict(r)
            nr["_conf_channels"] = pchannels
            best[k] = nr
        else:
            prev["_conf_channels"] = pchannels
    return sorted(best.values(), key=lambda row: (-float(row["confidence"]), str(row["skill"])))


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
        if not (_is_skill_heading(sec) or _is_certification_heading(sec)):
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
        calibratable_source = (
            source.startswith("exact")
            or source.startswith("fuzzy")
            or source == "synonym"
            or source.startswith("semantic:")
            or source.startswith("ner_span")
        )
        if cal_on and calibratable_source:
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
    s = re.sub(r"\|+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;|/\\-")
    s = _EVIDENCE_NOISE_LEAD_RE.sub("", s).strip()
    s = re.sub(r"^[a-z]\s+(?=[a-z0-9]{3,})", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(including|strong|advanced)\b$", "", s, flags=re.IGNORECASE).strip(" .,:;|/\\-")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _clean_evidence_item(item: str) -> str:
    s = str(item or "").strip()
    if not s:
        return ""
    if s.startswith("section:"):
        return s
    s = _trim_evidence_fragment(s)
    if not s:
        return ""
    low = _normalize_text(s)
    if low in {"strong", "including", "advanced", "n/a", "na"}:
        return ""
    return s


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
        cleaned = _clean_evidence_item(chunk or "")
        if cleaned and cleaned not in evidence:
            evidence.append(cleaned)
        if len(evidence) >= max_items:
            break
    return evidence


def _header_lines(text: str, limit: int = 24) -> list[str]:
    lines = [
        re.sub(r"\s+", " ", ln[:MAX_LINE_CHARS]).strip()
        for ln in (text or "").splitlines()[:MAX_LINES]
        if ln and ln.strip()
    ]
    return lines[:limit]


def _normalize_name_line(raw_line: str) -> str:
    line = re.sub(r"\s+", " ", str(raw_line or "")).strip()
    if not line:
        return ""
    line = re.split(r"\s*•\s*|\s+\|\s+", line, maxsplit=1)[0]
    return line.strip(" -,:;|")


def _derive_name_from_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    local_part = str(email).split("@", 1)[0].strip()
    if not local_part:
        return None
    if re.search(r"[^a-zA-Z._-]", local_part):
        return None
    tokens = [tok for tok in re.split(r"[._-]+", local_part) if tok]
    if not 2 <= len(tokens) <= 4:
        return None
    candidate = " ".join(tok.capitalize() for tok in tokens)
    return candidate if _looks_like_person_name(candidate) else None


def _looks_like_person_name(line: str) -> bool:
    line = _normalize_name_line(line)
    if not line or len(line) < 5 or len(line) > 80:
        return False
    lowered = _normalize_text(line).replace("\n", " ").strip()
    if not lowered:
        return False
    if any(hint in lowered for hint in _CONTACT_URL_HINTS):
        return False
    if _EMAIL_RE.search(line) or _PHONE_RE.search(line) or re.search(r"\d", line):
        return False
    is_heading, _key, _weight = _heading_candidate(line)
    if is_heading:
        return False

    tokens = [tok.strip(".,;:()[]{}") for tok in line.split() if tok.strip(".,;:()[]{}")]
    if not 2 <= len(tokens) <= 5:
        return False
    if any(not _NAME_TOKEN_RE.fullmatch(tok) for tok in tokens):
        return False

    lowered_tokens = [tok.lower() for tok in tokens]
    if any(tok in _NAME_STOPWORDS for tok in lowered_tokens):
        return False
    if any(keyword in lowered for keyword in _TITLE_KEYWORDS):
        return False

    capitalized = sum(1 for tok in tokens if tok[:1].isupper() or tok.isupper())
    return capitalized >= max(2, len(tokens) - 1)


def extract_full_name(text: str, email_hint: str | None = None) -> str | None:
    for line in _header_lines(text, limit=12):
        normalized = _normalize_name_line(line)
        if _looks_like_person_name(normalized):
            return normalized
    inferred_from_email = _derive_name_from_email(email_hint)
    if inferred_from_email:
        return inferred_from_email
    return None


def _contact_text_variants(text: str) -> list[str]:
    variants: list[str] = []

    def _append(value: str) -> None:
        candidate = str(value or "").strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    base = str(text or "")[:_CONTACT_SCAN_CHARS]
    _append(base)

    header = "\n".join(_header_lines(base, limit=40))
    if header and header != base:
        _append(header)

    seeds = list(variants)
    for seed in seeds:
        normalized = _OBFUSCATED_AT_RE.sub("@", seed)
        normalized = _OBFUSCATED_DOT_RE.sub(".", normalized)
        normalized = re.sub(r"\s*@\s*", "@", normalized)
        normalized = re.sub(r"\s*\.\s*", ".", normalized)
        _append(normalized)

        compact = re.sub(r"[\s\u200b\u200c\u200d]+", "", normalized)
        _append(compact)

    return variants


def extract_email(text: str) -> str | None:
    for variant in _contact_text_variants(text):
        match = _EMAIL_RE.search(variant)
        if not match:
            continue
        normalized = match.group(0).strip().lower()
        if normalized:
            return normalized
    match = _EMAIL_RE.search(text)
    if match:
        normalized = match.group(0).strip().lower()
        if normalized:
            return normalized
    fallback_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}", text, re.IGNORECASE)
    if fallback_match:
        normalized = fallback_match.group(0).strip().lower()
        if normalized:
            return normalized
    return None


def _clean_phone_candidate(value: str) -> str | None:
    cleaned = str(value or "")
    cleaned = cleaned.replace("\u200b", " ").replace("\u200c", " ").replace("\u200d", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,:;|.")
    if _DATE_RANGE_RE.match(cleaned):
        return None
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 8 or len(digits) > 15:
        return None
    if len(set(digits)) <= 1:
        return None
    if "/" in cleaned and len(digits) == 8:
        return None
    return cleaned or None


def extract_phone(text: str) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(_header_lines(text, limit=40)):
        line_lower = line.lower()
        for match in _PHONE_RE.finditer(line):
            cleaned = _clean_phone_candidate(match.group(0))
            if not cleaned:
                continue
            score = 0
            if cleaned.startswith("+"):
                score += 3
            if any(hint in line_lower for hint in _CONTACT_LABEL_HINTS):
                score += 3
            if _EMAIL_RE.search(line):
                score += 2
            if index <= 4:
                score += 1
            candidates.append((score, -index, cleaned))
    if not candidates:
        for variant in _contact_text_variants(text):
            for match in _PHONE_RE.finditer(variant):
                cleaned = _clean_phone_candidate(match.group(0))
                if not cleaned:
                    continue
                context = variant[max(0, match.start() - 40): min(len(variant), match.end() + 40)].lower()
                score = 0
                if cleaned.startswith("+"):
                    score += 2
                if any(hint in context for hint in _CONTACT_LABEL_HINTS):
                    score += 3
                if "@" in context:
                    score += 1
                candidates.append((score, -match.start(), cleaned))
    if not candidates:
        for match in _PHONE_RE.finditer(text):
            cleaned = _clean_phone_candidate(match.group(0))
            if not cleaned:
                continue
            candidates.append((0, -match.start(), cleaned))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def extract_contact_details(text: str) -> dict[str, str | None]:
    email = extract_email(text)
    return {
        "full_name": extract_full_name(text, email_hint=email),
        "email": email,
        "phone": extract_phone(text),
    }


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


def _build_certification_exclusion_keys(certifications: list[str]) -> set[str]:
    keys: set[str] = set()
    for raw in certifications or []:
        text = str(raw or "").strip()
        if not text:
            continue
        ckey = _skill_key(text)
        if ckey:
            keys.add(ckey)
        normalized = _normalize_for_pattern(text)
        if not normalized:
            continue
        for match in _CERTIFICATION_ACRONYM_RE.findall(normalized):
            mkey = _skill_key(match)
            if mkey:
                keys.add(mkey)
        for alias in _CERTIFICATION_ALIAS_HINTS:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                akey = _skill_key(alias)
                if akey:
                    keys.add(akey)
    return keys


def _skill_row_should_be_dropped(
    row: dict[str, Any],
    *,
    certification_keys: set[str],
) -> bool:
    raw_skill = str(row.get("skill", "")).strip()
    if not raw_skill:
        return True
    low = _normalize_for_pattern(raw_skill)
    sk = _skill_key(raw_skill)

    if not low or not sk:
        return True
    if low in _SKILL_SECTION_NOISE_TERMS:
        return True
    if low in _SKILL_LOCATION_NOISE_TERMS:
        return True
    if re.fullmatch(r"\d+", low):
        return True
    if _SKILL_LEADING_NUMBER_NOISE_RE.match(low):
        return True
    if _SKILL_CERT_SPLIT_NOISE_RE.match(low):
        return True
    if _looks_like_semantic_section_heading(low, low.split()):
        return True
    if sk in certification_keys:
        return True
    if _looks_like_certification_entry(raw_skill):
        return True
    return False


def _prune_skill_rows_with_context(rows: list[dict[str, Any]], certifications: list[str]) -> list[dict[str, Any]]:
    cert_keys = _build_certification_exclusion_keys(certifications)
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _skill_row_should_be_dropped(row, certification_keys=cert_keys):
            continue
        out.append(row)
    return out


def _apply_evidence_based_gating(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out skills without sufficient evidence, except for strong sources."""
    strong_sources = {"exact", "fuzzy", "synonym"}  
    out: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source", ""))
        evidence = row.get("evidence", [])
        conf = float(row.get("confidence", 0))
        if source.startswith("cv_section:") or source.startswith("ner_span:") or source.startswith("semantic_augment"):
            if not evidence and conf < 0.65:
                continue
        elif source.startswith("sentence_"):
            if not evidence:
                continue
        out.append(row)
    return out

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
            "certification": [],
            "hands_on_project": [],
            "project_text": [],
            "project_validated_skill": [],
        },
        "preview": "",
        "extracted_skills": [],
        "extracted_full_name": None,
        "extracted_email": None,
        "extracted_phone": None,
        "predicted_title": None,
        "predicted_experience_years": None,
        "certifications": [],
        "hands_on_projects": [],
        "project_skill_links": [],
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
    result["preview"] = text[:_PREVIEW_CONTACT_CHARS]

    if not text.strip():
        result["degraded"] = True
        result["warnings"].append("empty_text")
        return result

    try:
        contact_details = extract_contact_details(text)
        result["extracted_full_name"] = contact_details.get("full_name")
        result["extracted_email"] = contact_details.get("email")
        result["extracted_phone"] = contact_details.get("phone")
        if not result["extracted_email"] and "@" in text:
            result["warnings"].append("email_regex_no_match")
        if not result["extracted_phone"] and re.search(r"\d{7,}", text):
            result["warnings"].append("phone_regex_no_match")
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("extract_contact_details", exc))

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
        certifications = extract_certifications_from_sections(text)
        hands_on_projects = extract_hands_on_projects_from_sections(text)
        result["certifications"] = certifications
        result["hands_on_projects"] = hands_on_projects
        result["extraction_channels"]["certification"] = list(certifications)
        result["extraction_channels"]["hands_on_project"] = list(hands_on_projects)
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("extract_candidate_boards", exc))

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
        rejected_project_phrases: list[str] = []
        open_rows: list[dict[str, Any]] = []
        ner_rows: list[dict[str, Any]] = []
        soft_rows: list[dict[str, Any]] = []
        sentence_rows: list[dict[str, Any]] = []
        context_board_rows: list[dict[str, Any]] = []

        cv_ner_rows: list[dict[str, Any]] = []
        try:
            from app.services.cv_ner_inference import extract_skills_filtered, is_available
            if is_available():
                cv_ner_rows = extract_skills_filtered(text)
            else:
                result["warnings"].append("cv_ner_model_unavailable")
        except Exception as exc:
            result["warnings"].append("cv_ner_model_unavailable")
            result["errors"].append(_stage_error("cv_ner_finetuned", exc))

        rows = _merge_skill_rows(rows, cv_ner_rows)
        soft_channel_names: list[str] = []
        rows = enrich_skill_confidence_rows(rows, text=text, sections=section_map)
        rows_with_evidence: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("skill"):
                continue
            enhanced = dict(row)
            prior_ev = list(enhanced.get("evidence") or [])
            collected = _collect_skill_evidence(str(row.get("skill", "")), text=text)
            cleaned_evidence: list[str] = []
            for ev in prior_ev + collected:
                c_ev = _clean_evidence_item(str(ev or ""))
                if c_ev and c_ev not in cleaned_evidence:
                    cleaned_evidence.append(c_ev)
            enhanced["evidence"] = cleaned_evidence[:3]
            src = str(enhanced.get("source", ""))
            try:
                conf_val = float(enhanced.get("confidence", 0.0))
            except Exception:
                conf_val = 0.0
            enhanced["source_label"] = _source_label_key(src)
            enhanced["confidence_band"] = _confidence_band_for_source(src, conf_val)
            rows_with_evidence.append(enhanced)
        rows = rows_with_evidence
        apply_context_strength_to_rows(rows)
        rows = _apply_evidence_based_gating(rows)
        apply_weak_hedge_penalty_to_rows(rows)
        attach_confidence_normalized(rows)
        rows, dropped_by_source_gate = apply_post_merge_source_gate(rows, min_confidence=safe_min_conf)
        if dropped_by_source_gate:
            result["warnings"].append(f"post_merge_source_gate_dropped:{dropped_by_source_gate}")
        pre_prune_count = len(rows)
        rows = _prune_skill_rows_with_context(
            rows,
            certifications=[str(item or "") for item in result.get("certifications", [])],
        )
        dropped_by_context = max(0, pre_prune_count - len(rows))
        if dropped_by_context:
            result["warnings"].append(f"context_skill_prune_dropped:{dropped_by_context}")

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
        project_links = build_project_skill_links(
            projects=[str(item or "") for item in result.get("hands_on_projects", [])],
            extracted_rows=rows,
        )
        result["project_skill_links"] = project_links
        result["extraction_channels"]["project_validated_skill"] = sorted(
            {str(link.get("skill", "")).strip() for link in project_links if str(link.get("skill", "")).strip()},
            key=lambda x: x.lower(),
        )
        if not result["skills"]:
            result["warnings"].append("no_skills_detected")
        if budget_hit:
            result["warnings"].append("skills_time_budget_hit")
            if not result["skills"]:
                result["degraded"] = True
    except Exception as exc:
        result["degraded"] = True
        result["errors"].append(_stage_error("detect_skills_with_confidence", exc))
        try:
            legacy = detect_skills(text=text, known_skills=skills_list)
            result["skills"] = [str(s).strip() for s in legacy if isinstance(s, str) and str(s).strip()]
            legacy_rows = [
                {"skill": s, "confidence": 0.6, "source": "legacy", "evidence": []} for s in result["skills"]
            ]
            apply_weak_hedge_penalty_to_rows(legacy_rows)
            attach_confidence_normalized(legacy_rows)
            for row in legacy_rows:
                src = str(row.get("source", "legacy"))
                row["source_label"] = _source_label_key(src)
                row["confidence_band"] = _confidence_band_for_source(src, float(row.get("confidence", 0.0)))
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
