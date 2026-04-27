"""
Convert the 155 Ollama-generated CV .txt files into labeled training JSONL.

The original generation script had a bug: each .txt file contains the raw
Ollama JSON envelope (e.g. {"model":..., "response":"<cv text>", "done":true}),
not just the CV text. This script extracts the actual CV body and runs a
markdown-aware auto-labeler tuned for the format the LLM produced:

  **Certifications:**
  * Cert Name, Institution (YYYY)
  * ...

  **Projects:**
  * **Project Name**: Description...
  * ...

Output: data/labels/cv_extraction_ollama_155_extended.jsonl with fields:
    {"id", "text", "labels": {"certifications": [...], "projects": [...]}, "meta": {...}}

This file plugs into the existing BIO-conversion pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "Cert-CV" / "cert CV"
OUTPUT_FILE = ROOT / "data" / "labels" / "cv_extraction_ollama_155_extended.jsonl"

# Section headings (markdown asterisks already stripped before lookup)
CERT_HEADINGS = {
    "certifications", "certificates", "certification", "certificats",
    "licenses", "licenses and certifications",
}
PROJECT_HEADINGS = {
    "projects", "key projects", "selected projects", "personal projects",
    "academic projects", "professional projects", "portfolio",
    "projets", "projets realises",
}
OTHER_HEADINGS = {
    "education", "experience", "work experience", "professional experience",
    "skills", "technical skills", "competences", "summary", "profile",
    "references", "interests", "languages", "contact information",
    "objective", "achievements",
}
ALL_HEADINGS = CERT_HEADINGS | PROJECT_HEADINGS | OTHER_HEADINGS

YEAR_PARENS_RE = re.compile(r"\(\s*(?:19|20)\d{2}\s*\)\s*$")
BULLET_PREFIX_RE = re.compile(r"^[\-\*•·∙▪►–—\+]\s+")


def clean_markdown(text: str) -> str:
    """Strip markdown formatting that confuses heading/entity detection."""
    # Bold: **text** → text
    text = re.sub(r"\*\*([^*]+?)\*\*", r"\1", text)
    # Italic underscore: _text_ → text (avoid stripping underscores in identifiers)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)
    # Markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Headings (## Foo → Foo)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    return text


def extract_cv_text(file_path: Path) -> str | None:
    """Each .txt is actually an Ollama JSON envelope. Extract `.response`."""
    try:
        raw = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"  WARN: read failed {file_path.name}: {exc}", file=sys.stderr)
        return None
    raw = raw.strip()
    # Some files might already be plain text (if user retried)
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        cv = obj.get("response") or ""
    else:
        cv = raw
    cv = cv.strip()
    return cv if len(cv) >= 200 else None


def is_heading(line: str) -> str | None:
    """Return the lowercased heading key if line looks like a section heading."""
    s = line.strip().rstrip(":").strip()
    if not s or len(s.split()) > 5:
        return None
    if not (s.isupper() or s[0].isupper()):
        return None
    low = s.lower()
    if low in ALL_HEADINGS:
        return low
    for h in ALL_HEADINGS:
        if low == h or low.startswith(h + " "):
            return h
    return None


def classify(heading_key: str) -> str:
    if heading_key in CERT_HEADINGS:
        return "cert"
    for h in CERT_HEADINGS:
        if heading_key.startswith(h):
            return "cert"
    if heading_key in PROJECT_HEADINGS:
        return "project"
    for h in PROJECT_HEADINGS:
        if heading_key.startswith(h):
            return "project"
    return "other"


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    cur_type = "intro"
    cur: list[str] = []
    for raw in text.splitlines():
        h = is_heading(raw)
        if h is not None:
            sections.append((cur_type, cur))
            cur_type = classify(h)
            cur = []
        else:
            cur.append(raw)
    sections.append((cur_type, cur))
    return sections


def extract_cert_entries(lines: list[str]) -> list[str]:
    """
    Cert section format: bullet-prefixed entries ending in (YYYY).
    Example: "* Certified Analytics Professional (CAP), INFORMS (2020)"
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        bare = BULLET_PREFIX_RE.sub("", line).strip()
        if not bare:
            continue
        # Must end with (YYYY) — the strongest cert signal
        if not YEAR_PARENS_RE.search(bare):
            continue
        # Reject overly long lines (>20 words = probably a description that mentions a year)
        if len(bare.split()) > 20:
            continue
        key = bare.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(bare)
    return out


def extract_project_entries(lines: list[str]) -> list[str]:
    """
    Project section format: bullet-prefixed entries with name before colon.
    Example: "* Customer Segmentation Analysis: Developed a clustering model..."
    We extract the name (text before the first ":") if it looks valid.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        bare = BULLET_PREFIX_RE.sub("", line).strip()
        if not bare:
            continue
        # Take part before the first colon (or hyphen separator) as project name
        # but only if there's a description after — otherwise keep whole line
        name = bare
        for sep in (": ", " - ", " – ", " — "):
            if sep in bare:
                head = bare.split(sep, 1)[0].strip()
                if head and head != bare:
                    name = head
                    break
        # Strip leading/trailing punctuation
        name = name.strip(" .,:;|")
        if not name:
            continue
        words = name.split()
        # 1-8 words; first letter must be uppercase (Title Case or ALL CAPS)
        if not (1 <= len(words) <= 8):
            continue
        if not name[0].isupper():
            continue
        # Reject if the name itself looks like a sentence (contains action verbs)
        if re.search(
            r"\b(?:developed|built|created|designed|implemented|managed|led|"
            r"deployed|conducted|contributed|maintained|migrated|optimized|"
            r"developpe|cree|concu|gere)\b",
            name, re.IGNORECASE,
        ):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def label_cv(text: str) -> dict:
    cleaned = clean_markdown(text)
    certs: list[str] = []
    projects: list[str] = []
    for sec_type, lines in split_sections(cleaned):
        if sec_type == "cert":
            certs.extend(extract_cert_entries(lines))
        elif sec_type == "project":
            projects.extend(extract_project_entries(lines))
    # Validate that labels appear in the ORIGINAL text (not the markdown-cleaned
    # version) so BIO conversion can find them. Drop entries that no longer match.
    text_for_match = text
    certs_kept = [c for c in certs if c in text_for_match or c in cleaned]
    projects_kept = [p for p in projects if p in text_for_match or p in cleaned]
    return {
        "certifications": certs_kept,
        "projects": projects_kept,
        "cleaned_text": cleaned,
    }


def main():
    if not INPUT_DIR.exists():
        print(f"ERROR: input dir not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"Found {len(files)} .txt files in {INPUT_DIR}")

    stats = Counter()
    samples: list[dict] = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for fp in files:
            text = extract_cv_text(fp)
            if not text:
                stats["skipped_no_text"] += 1
                continue
            labeled = label_cv(text)
            certs = labeled["certifications"]
            projects = labeled["projects"]
            cleaned_text = labeled["cleaned_text"]

            stats["records"] += 1
            if certs:
                stats["with_certs"] += 1
                stats["total_certs"] += len(certs)
            if projects:
                stats["with_projects"] += 1
                stats["total_projects"] += len(projects)
            if not (certs or projects):
                stats["records_with_no_labels"] += 1

            # Use the markdown-cleaned text so labels align cleanly during BIO conversion
            record = {
                "id": fp.stem,
                "text": cleaned_text,
                "labels": {
                    "certifications": certs,
                    "projects": projects,
                },
                "meta": {
                    "source": "ollama_local_legacy",
                    "model": "llama3",
                    "filename": fp.name,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (certs or projects) and len(samples) < 5:
                samples.append({
                    "id": fp.stem,
                    "certs": certs,
                    "projects": projects,
                })

    print()
    print(f"Records written:           {stats['records']}")
    print(f"Records with cert labels:  {stats['with_certs']}  (total cert entities: {stats['total_certs']})")
    print(f"Records with project labels: {stats['with_projects']}  (total project entities: {stats['total_projects']})")
    print(f"Records with NO labels:    {stats['records_with_no_labels']}")
    print(f"Skipped (no text):         {stats['skipped_no_text']}")
    print()
    print("Sample labels for spot-check:")
    for s in samples:
        print(f"\n[{s['id']}]")
        if s["certs"]:
            print(f"  certs:    {s['certs']}")
        if s["projects"]:
            print(f"  projects: {s['projects']}")
    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
