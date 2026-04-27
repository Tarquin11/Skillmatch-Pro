"""
Auto-label certifications and projects in existing CV training data using
high-precision structural heuristics.

The heuristics here are used ONCE during data preparation to bootstrap labels
for NER training. They are NOT used at inference time. The trained model
learns from the patterns and generalizes beyond what these rules cover.

Conservative principle: prefer false negatives over false positives. The
model will generalize from clean examples; corrupt labels permanently teach
the model wrong patterns.

Reads:
  - data/labels/cv_extraction_synth_200.jsonl   (200 synth records, inline text)
  - data/labels/cv_extraction_hf_labels.jsonl   (800 HF records, docx-backed)
  - data/cv_hf_docs/                             (the docx files)

Writes:
  - data/labels/cv_extraction_synth_200_extended.jsonl
  - data/labels/cv_extraction_hf_labels_extended.jsonl

Each output record gets new fields:
  - certifications: list of cert name strings
  - projects:       list of project name strings
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
SYNTH_IN  = ROOT / "data" / "labels" / "cv_extraction_synth_200.jsonl"
HF_IN     = ROOT / "data" / "labels" / "cv_extraction_hf_labels.jsonl"
HF_DOCS   = ROOT / "data" / "cv_hf_docs"
SYNTH_OUT = ROOT / "data" / "labels" / "cv_extraction_synth_200_extended.jsonl"
HF_OUT    = ROOT / "data" / "labels" / "cv_extraction_hf_labels_extended.jsonl"

# ---------------- heading vocabulary (label-time only) -------------------

CERT_HEADINGS = {
    # English
    "certifications", "certificate", "certificates", "certifications obtained",
    "professional certifications", "licenses", "licenses and certifications",
    # French — only TRUE cert headings, not "formation" (= education) or
    # "diplome" (= diploma, ambiguous). False positives there poisoned labels.
    "certificats", "certificat", "certifs",
    "certifications obtenues", "certifications professionnelles",
    "attestations", "attestation",
    "formations et certifications",
}
PROJECT_HEADINGS = {
    "projects", "personal projects", "academic projects",
    "professional projects", "portfolio", "side projects",
    "hands on projects", "hands-on projects",
    "projets", "projets realises", "projets pratiques",
    "mes projets", "travaux", "realisations",
    "applications developpees",
}
OTHER_HEADINGS = {
    "education", "experience", "work experience", "professional experience",
    "skills", "competences", "competences techniques", "technical skills",
    "languages", "langues", "interests", "centres d'interet", "loisirs",
    "summary", "profile", "objective", "references", "publications",
    "awards", "achievements", "vie associative", "associative life",
    "contact", "coordonnees", "about", "a propos",
    # French training/education that are NOT cert sections
    "formation", "formations", "diplome", "diplomes",
    "training", "trainings", "cours", "courses",
}
ALL_HEADINGS = CERT_HEADINGS | PROJECT_HEADINGS | OTHER_HEADINGS

# ---------------- entity-shape regexes (label-time only) -----------------

YEAR_PARENS_RE = re.compile(r"\(\s*(?:19|20)\d{2}\s*\)\s*$")
CERT_ACRONYM_RE = re.compile(
    r"\b(?:AWS|Azure|GCP|PMP|ITIL|CISA|CISSP|CCNA|CCNP|CompTIA|"
    r"Security\+|Network\+|MCSE|MCSA|MOS|TOEFL|IELTS|"
    r"OCA|OCP|RHCSA|RHCE|CKA|CKAD|TOGAF|PRINCE2)\b",
    re.IGNORECASE,
)
CERT_KEYWORDS_RE = re.compile(
    r"\b(?:certified|certificate|certification|certificat|"
    r"diploma|diplome|attestation)\b",
    re.IGNORECASE,
)

ALL_CAPS_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9_-]{2,}$")
TITLE_CASE_LINE_RE = re.compile(
    r"^(?:[A-ZÀ-Ý][A-Za-zÀ-ÿ\-]*\s+){1,5}[A-ZÀ-Ý][A-Za-zÀ-ÿ\-]*\s*(?:\(\d{4}\))?$"
)
ACTION_VERB_RE = re.compile(
    r"\b(?:developed|developing|built|building|created|creating|"
    r"designed|designing|implemented|implementing|managed|managing|"
    r"led|leading|deployed|deploying|conducted|conducting|"
    r"contributed|maintained|migrated|optimized|"
    r"developpe|developpement|cree|creation|concu|conception|"
    r"gere|gestion|realise|implemente|deploye)\b",
    re.IGNORECASE,
)
BULLET_PREFIX_RE = re.compile(r"^[-*•·∙▪►–—]\s*")

MAX_CERT_WORDS    = 12
MAX_PROJECT_WORDS = 8
MAX_HEADING_WORDS = 5


# ---------------- helpers ---------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def normalize_heading_key(line: str) -> str | None:
    """If `line` looks like a section heading, return its lowercased,
    accent-stripped form; else None."""
    s = line.strip().rstrip(":").strip()
    if not s:
        return None
    words = s.split()
    if len(words) > MAX_HEADING_WORDS:
        return None
    # Heading should be all-caps OR title-case (not a regular sentence).
    if not (s.isupper() or s[0].isupper()):
        return None
    low = _strip_accents(s.lower())
    if low in ALL_HEADINGS:
        return low
    # Compound match (e.g., "projets realises personnels")
    for h in ALL_HEADINGS:
        if low.startswith(h + " ") or low == h:
            return h
    return None


def classify_heading(key: str) -> str:
    if key in CERT_HEADINGS:
        return "cert"
    for h in CERT_HEADINGS:
        if key.startswith(h):
            return "cert"
    if key in PROJECT_HEADINGS:
        return "project"
    for h in PROJECT_HEADINGS:
        if key.startswith(h):
            return "project"
    return "other"


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Yield (section_type, lines) tuples. section_type ∈ {intro, cert, project, other}."""
    sections: list[tuple[str, list[str]]] = []
    current_type = "intro"
    current: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        key = normalize_heading_key(line)
        if key is not None:
            sections.append((current_type, current))
            current_type = classify_heading(key)
            current = []
        else:
            current.append(line)
    sections.append((current_type, current))
    return sections


def is_cert_entity(line: str) -> bool:
    s = BULLET_PREFIX_RE.sub("", line).strip()
    if not s or len(s.split()) > MAX_CERT_WORDS:
        return False
    has_year_parens = bool(YEAR_PARENS_RE.search(s))
    has_acronym     = bool(CERT_ACRONYM_RE.search(s))
    has_keyword     = bool(CERT_KEYWORDS_RE.search(s))
    return has_year_parens or has_acronym or has_keyword


def is_project_entity(line: str) -> bool:
    s = BULLET_PREFIX_RE.sub("", line).strip()
    if not s:
        return False
    words = s.split()
    if len(words) > MAX_PROJECT_WORDS:
        return False
    if ACTION_VERB_RE.search(s):
        return False  # full-sentence description, not a project name
    base = re.sub(r"\s*\(\d{4}\)\s*$", "", s).strip()
    if not base:
        return False
    has_year_parens = bool(YEAR_PARENS_RE.search(s))
    is_all_caps = base.isupper() and len(base.replace(" ", "")) >= 3
    is_title_case = bool(TITLE_CASE_LINE_RE.match(s))
    return has_year_parens or is_all_caps or is_title_case


def label_section_lines(lines: list[str], detect_fn) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        bare = BULLET_PREFIX_RE.sub("", line).strip()
        if not bare:
            continue
        if not detect_fn(bare):
            continue
        key = bare.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(bare)
    return out


def auto_label_text(text: str) -> dict:
    certs: list[str] = []
    projects: list[str] = []
    for sec_type, lines in split_sections(text):
        if sec_type == "cert":
            certs.extend(label_section_lines(lines, is_cert_entity))
        elif sec_type == "project":
            projects.extend(label_section_lines(lines, is_project_entity))
    return {"certifications": certs, "projects": projects}


# ---------------- main pipelines -------------------------------------------

def process_synth() -> dict:
    if not SYNTH_IN.exists():
        print(f"  WARN: {SYNTH_IN} missing", file=sys.stderr)
        return {"records": 0, "with_certs": 0, "with_projects": 0,
                "total_certs": 0, "total_projects": 0, "samples": []}
    stats = Counter()
    samples = []

    with SYNTH_IN.open(encoding="utf-8") as fin, \
         SYNTH_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            new = auto_label_text(text)
            labels = obj.get("labels") or {}
            labels["certifications"] = new["certifications"]
            labels["projects"] = new["projects"]
            obj["labels"] = labels
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["records"] += 1
            if new["certifications"]:
                stats["with_certs"] += 1
                stats["total_certs"] += len(new["certifications"])
            if new["projects"]:
                stats["with_projects"] += 1
                stats["total_projects"] += len(new["projects"])
            if (new["certifications"] or new["projects"]) and len(samples) < 8:
                samples.append({
                    "id": obj.get("id"),
                    "certs": new["certifications"],
                    "projects": new["projects"],
                })
    return {
        "records": int(stats["records"]),
        "with_certs": int(stats["with_certs"]),
        "with_projects": int(stats["with_projects"]),
        "total_certs": int(stats["total_certs"]),
        "total_projects": int(stats["total_projects"]),
        "samples": samples,
    }


def process_hf() -> dict:
    if not HF_IN.exists():
        print(f"  WARN: {HF_IN} missing", file=sys.stderr)
        return {"records": 0, "with_certs": 0, "with_projects": 0,
                "total_certs": 0, "total_projects": 0, "samples": []}
    stats = Counter()
    samples = []

    with HF_IN.open(encoding="utf-8") as fin, \
         HF_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            path_str = obj.get("path", "") or ""
            name = path_str.replace("\\", "/").split("/")[-1] if path_str else ""
            if not name:
                continue
            if not name.endswith(".docx"):
                name = name + ".docx"
            local = HF_DOCS / name
            if not local.exists():
                continue
            try:
                doc = Document(str(local))
                text = "\n".join(p.text for p in doc.paragraphs if p.text)
            except Exception as exc:
                print(f"  WARN: failed reading {local.name}: {exc}", file=sys.stderr)
                continue
            new = auto_label_text(text)
            obj["text"] = text  # store text inline so BIO converter can reuse
            obj["certifications"] = new["certifications"]
            obj["projects"] = new["projects"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["records"] += 1
            if new["certifications"]:
                stats["with_certs"] += 1
                stats["total_certs"] += len(new["certifications"])
            if new["projects"]:
                stats["with_projects"] += 1
                stats["total_projects"] += len(new["projects"])
            if (new["certifications"] or new["projects"]) and len(samples) < 8:
                samples.append({
                    "id": local.stem,
                    "certs": new["certifications"],
                    "projects": new["projects"],
                })
    return {
        "records": int(stats["records"]),
        "with_certs": int(stats["with_certs"]),
        "with_projects": int(stats["with_projects"]),
        "total_certs": int(stats["total_certs"]),
        "total_projects": int(stats["total_projects"]),
        "samples": samples,
    }


def main():
    print("Auto-labeling cert/project entities (Option-B heuristic pass)")
    print(f"  synth in:  {SYNTH_IN}")
    print(f"  hf in:     {HF_IN}")
    print(f"  hf docs:   {HF_DOCS}")
    print()

    print("Processing synth-200...")
    s_stats = process_synth()
    print(f"  records: {s_stats['records']}")
    print(f"  with cert labels:    {s_stats['with_certs']:>4}  "
          f"(total cert entities: {s_stats['total_certs']})")
    print(f"  with project labels: {s_stats['with_projects']:>4}  "
          f"(total project entities: {s_stats['total_projects']})")
    print()

    print("Processing HF-800 (this loads docx files, slower)...")
    h_stats = process_hf()
    print(f"  records: {h_stats['records']}")
    print(f"  with cert labels:    {h_stats['with_certs']:>4}  "
          f"(total cert entities: {h_stats['total_certs']})")
    print(f"  with project labels: {h_stats['with_projects']:>4}  "
          f"(total project entities: {h_stats['total_projects']})")
    print()

    print("=" * 60)
    print("SAMPLE LABELS — review for quality before training:")
    print("=" * 60)
    print("\n--- synth samples ---")
    for s in s_stats.get("samples", []):
        print(f"\n[{s['id']}]")
        if s["certs"]:
            print(f"  certs:    {s['certs']}")
        if s["projects"]:
            print(f"  projects: {s['projects']}")
    print("\n--- HF samples ---")
    for s in h_stats.get("samples", []):
        print(f"\n[{s['id']}]")
        if s["certs"]:
            print(f"  certs:    {s['certs']}")
        if s["projects"]:
            print(f"  projects: {s['projects']}")

    print()
    print(f"Output:")
    print(f"  {SYNTH_OUT}")
    print(f"  {HF_OUT}")


if __name__ == "__main__":
    main()
