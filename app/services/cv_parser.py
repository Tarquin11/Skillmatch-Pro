import io 
import re
from difflib import SequenceMatcher
import pdfplumber
from typing import Any, Iterable
from app.ai.preprocessing import normalize_skill_name

WORD_RE = re.compile(r"[a-z0-9+.#/\-]+")

def extract_text(file_bytes, filename):
    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
            return text
    return ""
    
def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "",(text or "").lower()).strip()

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

def detect_skills_with_confidence(
    text: str,
    known_skills: Iterable[str],
    min_confidence: float = 0.6,
) -> list[dict[str, Any]]:
    index = _build_skill_index(known_skills)
    if not index:
        return []
    normalized_text = _normalize_text(text)
    tokens = _tokenize(normalized_text)
    if not tokens:
        return []
    ngrams = _ngram_candidates(tokens, max_n=4)
    ngram_keys = {g: _skill_key(g) for g in ngrams}
    ngram_acronyms = {g: _acronym(g) for g in ngrams}
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