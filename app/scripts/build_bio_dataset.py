"""
Build a BIO-tagged training dataset from labeled CVs.

Reads two label sources:
  1. data/labels/cv_extraction_synth_200.jsonl  (inline text + labels, synthetic)
  2. data/labels/cv_extraction_hf_labels.jsonl  (paths + labels; text loaded from docx)

For each CV, aligns the entity labels (skills, tools, title, languages) to
token spans in the source text and emits BIO-tagged tokens. The output is
ready for HuggingFace token-classification fine-tuning.

OCR-aware matching is used for "very_dirty" synthetic CVs (handles digit-letter
substitutions like "D0ck3R" → "Docker").

Output: data/training/bio_dataset.jsonl
Each line: {"id", "source", "tokens": [...], "tags": [...]}

Usage:
    /path/to/venv/bin/python -m app.scripts.build_bio_dataset
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
SYNTH_LABELS = ROOT / "data" / "labels" / "cv_extraction_synth_200.jsonl"
HF_LABELS = ROOT / "data" / "labels" / "cv_extraction_hf_labels.jsonl"
OLLAMA_LABELS = ROOT / "data" / "labels" / "cv_extraction_ollama_155_extended.jsonl"
HF_DOCS_DIR = ROOT / "data" / "cv_hf_docs"
OUTPUT_DIR = ROOT / "data" / "training"
OUTPUT_FILE = OUTPUT_DIR / "bio_dataset.jsonl"

# OCR character swap for fuzzy matching adversarial noise
_OCR_SWAP = str.maketrans({"0": "o", "1": "i", "3": "e", "5": "s", "7": "t", "@": "a", "$": "s"})

TOKEN_RE = re.compile(r"[\w'-]+|[^\s\w]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenization. Returns just the surface tokens."""
    return [m.group(0) for m in TOKEN_RE.finditer(text or "")]


def normalize_token(tok: str, ocr: bool = False) -> str:
    n = tok.lower()
    if ocr:
        n = n.translate(_OCR_SWAP)
    return n


def find_entity_spans(
    text_tokens: list[str],
    entity: str,
    ocr_aware: bool = False,
) -> list[tuple[int, int]]:
    """
    Find all token-aligned positions where `entity` appears in `text_tokens`.
    Returns list of (start, end) token indices (end-exclusive).
    """
    entity_tokens = tokenize(entity)
    if not entity_tokens:
        return []

    norm_entity = [normalize_token(t, ocr=ocr_aware) for t in entity_tokens]
    norm_text = [normalize_token(t, ocr=ocr_aware) for t in text_tokens]
    n_e = len(norm_entity)
    n_t = len(norm_text)
    if n_e > n_t:
        return []

    spans: list[tuple[int, int]] = []
    for i in range(n_t - n_e + 1):
        if norm_text[i : i + n_e] == norm_entity:
            spans.append((i, i + n_e))
    return spans


def assign_bio_tags(
    num_tokens: int,
    spans: list[tuple[int, int, str]],
) -> list[str]:
    """
    Apply BIO tags. spans = list of (start, end, label).
    Conflict resolution: longer spans win; ties broken by earlier first.
    """
    tags = ["O"] * num_tokens
    occupied = [False] * num_tokens

    sorted_spans = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    for start, end, label in sorted_spans:
        if start < 0 or end > num_tokens:
            continue
        if any(occupied[start:end]):
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
        for i in range(start, end):
            occupied[i] = True
    return tags


def collect_entity_spans(
    text_tokens: list[str],
    entities: list[tuple[str, str]],
    ocr_aware: bool,
) -> list[tuple[int, int, str]]:
    """
    For each (entity_text, label) try to find spans in text_tokens.
    Returns combined list of (start, end, label) — possibly overlapping.
    """
    out: list[tuple[int, int, str]] = []
    for entity_text, label in entities:
        if not entity_text or not str(entity_text).strip():
            continue
        # Try exact (case-insensitive) first
        for s, e in find_entity_spans(text_tokens, entity_text, ocr_aware=False):
            out.append((s, e, label))
        # Then try OCR-aware if dirty
        if ocr_aware:
            for s, e in find_entity_spans(text_tokens, entity_text, ocr_aware=True):
                out.append((s, e, label))
    return out


def extract_entities_from_synth(labels: dict) -> list[tuple[str, str]]:
    """Map synth labels dict to list of (text, BIO_label) pairs."""
    out: list[tuple[str, str]] = []
    for s in labels.get("skills") or []:
        out.append((str(s), "SKILL"))
    for t in labels.get("tools") or []:
        out.append((str(t), "SKILL"))  # tools collapse into SKILL
    title = labels.get("title")
    if title:
        out.append((str(title), "TITLE"))
    for lang_obj in labels.get("languages") or []:
        if isinstance(lang_obj, dict):
            lang = lang_obj.get("language")
        else:
            lang = lang_obj
        if lang:
            out.append((str(lang), "LANGUAGE"))
    # Cert and project labels (added by process_ollama_certproject.py and the
    # auto_label_certs_projects.py step). Records that don't have these fields
    # contribute zero CERT/PROJECT spans, which is fine — they teach the model
    # that the rest of the CV is "O" for these classes.
    for c in labels.get("certifications") or []:
        out.append((str(c), "CERT"))
    for p in labels.get("projects") or []:
        out.append((str(p), "PROJECT"))
    return out


def extract_entities_from_hf(obj: dict) -> list[tuple[str, str]]:
    """Map HF labels (expected_skills, expected_title) to (text, BIO_label) pairs."""
    out: list[tuple[str, str]] = []
    for s in obj.get("expected_skills") or []:
        out.append((str(s), "SKILL"))
    title = obj.get("expected_title")
    if title:
        out.append((str(title), "TITLE"))
    return out


def load_docx_text(path: Path) -> str:
    try:
        doc = Document(str(path))
    except Exception as exc:
        print(f"  WARN: failed to read {path.name}: {exc}", file=sys.stderr)
        return ""
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def map_hf_path_to_local(win_path: str) -> Path | None:
    """HF labels store Windows paths; map to current repo's cv_hf_docs/ dir."""
    name = win_path.replace("\\", "/").split("/")[-1] if win_path else ""
    if not name:
        return None
    # Some manifests may omit the .docx extension.
    if not name.endswith(".docx"):
        name = f"{name}.docx"
    local = HF_DOCS_DIR / name
    return local if local.exists() else None


def process_synth_file(records_out: list, stats: Counter) -> None:
    if not SYNTH_LABELS.exists():
        print(f"  WARN: {SYNTH_LABELS} not found, skipping", file=sys.stderr)
        return
    with SYNTH_LABELS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["synth_json_errors"] += 1
                continue
            text = obj.get("text") or ""
            if not text:
                stats["synth_no_text"] += 1
                continue
            labels = obj.get("labels") or {}
            entities = extract_entities_from_synth(labels)
            if not entities:
                stats["synth_no_entities"] += 1
                continue

            ocr_aware = bool(obj.get("meta", {}).get("very_dirty", False))
            text_tokens = tokenize(text)
            if not text_tokens:
                stats["synth_no_tokens"] += 1
                continue

            spans = collect_entity_spans(text_tokens, entities, ocr_aware=ocr_aware)
            if not spans:
                stats["synth_no_alignment"] += 1
                continue

            tags = assign_bio_tags(len(text_tokens), spans)
            tagged = sum(1 for t in tags if t != "O")
            stats["synth_processed"] += 1
            stats["synth_tagged_tokens"] += tagged

            records_out.append({
                "id": str(obj.get("id") or ""),
                "source": "synth",
                "tokens": text_tokens,
                "tags": tags,
            })


def process_ollama_file(records_out: list, stats: Counter) -> None:
    """Process Ollama-generated CVs that have CERT and PROJECT labels."""
    if not OLLAMA_LABELS.exists():
        print(f"  WARN: {OLLAMA_LABELS} not found, skipping", file=sys.stderr)
        return
    with OLLAMA_LABELS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["ollama_json_errors"] += 1
                continue
            text = obj.get("text") or ""
            if not text:
                stats["ollama_no_text"] += 1
                continue
            labels = obj.get("labels") or {}
            entities = extract_entities_from_synth(labels)  # same schema
            if not entities:
                stats["ollama_no_entities"] += 1
                continue

            text_tokens = tokenize(text)
            if not text_tokens:
                stats["ollama_no_tokens"] += 1
                continue

            spans = collect_entity_spans(text_tokens, entities, ocr_aware=False)
            if not spans:
                stats["ollama_no_alignment"] += 1
                continue

            tags = assign_bio_tags(len(text_tokens), spans)
            tagged = sum(1 for t in tags if t != "O")
            stats["ollama_processed"] += 1
            stats["ollama_tagged_tokens"] += tagged

            records_out.append({
                "id": str(obj.get("id") or ""),
                "source": "ollama",
                "tokens": text_tokens,
                "tags": tags,
            })


def process_hf_file(records_out: list, stats: Counter) -> None:
    if not HF_LABELS.exists():
        print(f"  WARN: {HF_LABELS} not found, skipping", file=sys.stderr)
        return
    with HF_LABELS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["hf_json_errors"] += 1
                continue
            local = map_hf_path_to_local(obj.get("path") or "")
            if local is None:
                stats["hf_path_not_found"] += 1
                continue

            text = load_docx_text(local)
            if not text:
                stats["hf_no_text"] += 1
                continue

            entities = extract_entities_from_hf(obj)
            if not entities:
                stats["hf_no_entities"] += 1
                continue

            text_tokens = tokenize(text)
            if not text_tokens:
                stats["hf_no_tokens"] += 1
                continue

            spans = collect_entity_spans(text_tokens, entities, ocr_aware=False)
            if not spans:
                stats["hf_no_alignment"] += 1
                continue

            tags = assign_bio_tags(len(text_tokens), spans)
            tagged = sum(1 for t in tags if t != "O")
            stats["hf_processed"] += 1
            stats["hf_tagged_tokens"] += tagged

            records_out.append({
                "id": local.stem,
                "source": "hf",
                "tokens": text_tokens,
                "tags": tags,
            })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    stats: Counter = Counter()

    print("Processing synth labels...")
    process_synth_file(records, stats)
    print(f"  synth processed: {stats['synth_processed']}")

    print("Processing HF labels (loading docx files)...")
    process_hf_file(records, stats)
    print(f"  hf processed: {stats['hf_processed']}")

    print("Processing Ollama-generated cert/project labels...")
    process_ollama_file(records, stats)
    print(f"  ollama processed: {stats['ollama_processed']}")

    # Write JSONL
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Tag-distribution stats
    tag_counts: Counter = Counter()
    total_tokens = 0
    for rec in records:
        total_tokens += len(rec["tokens"])
        tag_counts.update(rec["tags"])

    print(f"\n{'=' * 50}")
    print(f"Total CVs in BIO dataset: {len(records)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"\nTag distribution:")
    for tag, count in tag_counts.most_common():
        pct = 100.0 * count / max(1, total_tokens)
        print(f"  {tag:<14} {count:>10,}  ({pct:5.2f}%)")
    print(f"\nDiagnostic stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<25} {v}")
    print(f"\nWritten to: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
