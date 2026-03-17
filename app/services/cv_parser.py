import io
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np
import pdfplumber
from docx import Document

from app.ai.preprocessing import normalize_skill_name
from app.services.embedding_service import EmbeddingService

WORD_RE = re.compile(r"[a-z0-9+.#/\-]+")
BULLET_PREFIXES = ("-", "*", "•", "â€¢")

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
    normalized = re.sub(r"[\u00a0\t\r\n]+", " ", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()

def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)

def _ngram_candidates(tokens: list[str], max_n: int = 4) -> set[str]:
    out = set(tokens)
    for n in range(2, max_n + 1):
        for i in range(0,max(len(tokens) -n + 1 , 0)):
            out.add(" ".join(tokens[i : i + n]))
    return out

def _skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]","", normalize_skill_name(value))

def _acronym(value: str) -> str:
    parts = [p for p in normalize_skill_name(value).split() if p]
    return "".join(p[0] for p in parts if p and p[0].isalnum())

def _build_skill_index(known_skills: Iterable[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for raw in known_skills or []:
        canonical = normalize_skill_name(raw)
        if not canonical:
            continue
        skill_key= _skill_key(canonical)
        if not skill_key:
            continue
        aliases = {skill_key}
        ac = _acronym(canonical)
        if len(ac) >=2:
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
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
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
            line = line.strip()
            if not line:
                continue
            for prefix in BULLET_PREFIXES:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            parts = re.split(r"[,\|;/]+", line)
            for part in parts:
                part = part.strip()
                if part:
                    phrases.append((part, section))
    return phrases

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

def detect_skills_with_confidence(
    text: str,
    known_skills: Iterable[str],
    min_confidence: float = 0.6,
    use_semantic: bool = True,
    semantic_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    known_list = [k for k in known_skills if k]
    index = _build_skill_index(known_list)
    if not index:
        return []
    normalized_text = _normalize_text(text)
    tokens = _tokenize(normalized_text)
    if not tokens:
        return []
    ngrams = _ngram_candidates(tokens, max_n=4)
    ngram_keys = {g: _skill_key(g) for g in ngrams}
    ngram_acronyms = {g: _acronym(g) for g in ngrams}
    sections, section_weights = _extract_sections(text)
    section_phrases = _extract_section_phrases(sections)
    skill_vectors, skill_names = _get_skill_embeddings(known_list) if use_semantic else (None, [])

    semantic_hits: dict[str, tuple[float, str]] = {}
    if use_semantic and skill_vectors is not None and skill_names:
        embedder = _get_embedder()
        if embedder:
            phrase_items: list[tuple[str, str]] = []
            seen_phrases: set[str] = set()
            for phrase, section in section_phrases:
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
                        idx = int(np.argmax(sims[i]))
                        sim = float(sims[i, idx])
                        if sim < semantic_threshold:
                            continue
                        skill = skill_names[idx]
                        weight = section_weights.get(section, 0.85)
                        conf = min(0.92, sim * weight)
                        prev = semantic_hits.get(skill)
                        if prev is None or conf > prev[0]:
                            semantic_hits[skill] = (conf, f"semantic:{section}")

    hits: list[dict[str, Any]] = []
    for canonical, meta in index.items():
        skill_key = meta["skill_key"]
        aliases = meta["aliases"]
        best_conf = 0.0
        best_source = "fuzzy"

        for gram in ngrams:
            gram_key = ngram_keys[gram]
            if not gram_key:
                continue
            if gram_key in aliases:
                conf = 0.98 if gram_key == skill_key else 0.90
                source = "exact" if gram_key == skill_key else "synonym"
            elif ngram_acronyms[gram] in aliases:
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

        for phrase, section in section_phrases:
            phrase_key = _skill_key(phrase)
            if not phrase_key:
                continue
            if phrase_key in aliases:
                base = 0.95
                conf = min(0.99, base * section_weights.get(section, 0.85))
                source = f"exact:{section}"
            else:
                ratio = SequenceMatcher(None, skill_key, phrase_key).ratio()
                if ratio < 0.88:
                    continue
                base = min(0.80, ratio)
                conf = min(0.90, base * section_weights.get(section, 0.85))
                source = f"fuzzy:{section}"
            if conf > best_conf:
                best_conf = conf
                best_source = source

        semantic = semantic_hits.get(canonical)
        if semantic and semantic[0] > best_conf:
            best_conf = semantic[0]
            best_source = semantic[1]

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

def detect_skills(text: str, known_skills: Iterable[str] | None = None) -> list[str]:
    if not known_skills:
        return []
    return [row["skill"] for row in detect_skills_with_confidence(text, known_skills)]
