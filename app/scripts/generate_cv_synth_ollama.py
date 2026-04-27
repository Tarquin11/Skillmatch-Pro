"""
Generate synthetic CVs with skill/title/cert/project labels using a local
Ollama model. Single LLM call per CV; one JSON object with text + entities.

Output: JSONL at data/labels/cv_extraction_synth_ollama.jsonl
Schema: same as data/labels/cv_extraction_synth_200.jsonl so the BIO
converter can process both with no changes.

Features:
  - Resumable: skips IDs already in output file
  - Diverse: rotates language (FR/EN), industry, experience level
  - Strict JSON output via Ollama's format=json mode (no parse failures)
  - Alignment validation: drops records where labeled entities don't appear
    verbatim in text (LLMs sometimes paraphrase)
  - Self-correcting: filters mis-aligned entities rather than rejecting whole CV

Usage:
    # Sanity check first — generates 5 CVs to verify pipeline + quality
    venv/bin/python -m app.scripts.generate_cv_synth_ollama --count 5

    # Full run (resumes if previous run was interrupted)
    venv/bin/python -m app.scripts.generate_cv_synth_ollama --count 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "labels" / "cv_extraction_synth_ollama.jsonl"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# --------------------------- diversity matrix ------------------------------

# Language weights — heavy on French because that's your real-world target.
LANGUAGES = [("fr", 0.6), ("en", 0.4)]

INDUSTRIES = [
    "software development", "data science", "machine learning", "web development",
    "mobile app development", "devops / cloud", "cybersecurity",
    "digital marketing", "product management", "ux/ui design",
    "finance", "accounting", "human resources", "sales",
    "healthcare", "education", "manufacturing engineering", "civil engineering",
    "legal", "consulting",
]
LEVELS = [
    "student / intern (0-1 years)",
    "junior (1-3 years)",
    "mid-level (3-6 years)",
    "senior (6-10 years)",
    "manager / lead (8+ years)",
]

LANG_FULL = {"en": "English", "fr": "French"}

PROMPT_TEMPLATE = """You are generating realistic synthetic CVs for an NER training dataset.

Generate ONE fictional CV in {language_full}.

Industry context: {industry}
Experience level: {level}

The CV must include these sections (use {language_full} section headings):
- Header (fictional name, email, phone, city)
- Job title or student status line
- SKILLS / COMPETENCES section with 5-10 concrete skills
- CERTIFICATIONS / CERTIFICATS section with 2-4 entries, EACH ending with a year in parentheses, e.g. "AWS Solutions Architect (2023)" or "Python pour la data science (2024)"
- PROJECTS / PROJETS section with 2-4 NAMED projects — each project has a short Title-Case or ALL-CAPS name (with optional year) followed by a 1-2 sentence description, e.g. "TaskFlow (2024) - A web app for team task management..."
- EXPERIENCE section with 1-3 past roles
- EDUCATION / FORMATION section

Output a JSON object with EXACTLY these fields and nothing else:
{{
  "text": "<the entire CV as one string, use \\n for line breaks>",
  "title": "<the current job title or student status>",
  "skills": ["<5-10 skills exactly as written in text>"],
  "certifications": ["<2-4 cert names exactly as written in text, INCLUDING the year in parens>"],
  "projects": ["<2-4 project names exactly as written in text — name only, NOT the description>"]
}}

CRITICAL RULES:
1. Every entry in skills/certifications/projects MUST appear VERBATIM in the text. Same case, same spelling, same punctuation.
2. Project names should be the SHORT identifier only (e.g. "TaskFlow" or "TaskFlow (2024)"), NOT the descriptive sentence after it.
3. Output ONLY the JSON object. No markdown fences, no commentary, no preamble.
"""


# --------------------------- Ollama interaction ----------------------------

def check_ollama() -> str | None:
    """Verify Ollama is reachable. Returns error message on failure, None on success."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        return None
    except Exception as exc:
        return str(exc)


def call_ollama(model: str, prompt: str, timeout: int = 180) -> dict | None:
    """One generation call. Returns the parsed JSON object the model emitted, or None."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Ollama forces valid JSON output
        "options": {
            "temperature": 0.85,
            "top_p": 0.95,
            "num_predict": 2048,
        },
    }
    try:
        r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as exc:
        print(f"    ollama call failed: {exc}", file=sys.stderr)
        return None
    body = r.json()
    out_str = body.get("response") or ""
    if not out_str:
        return None
    try:
        return json.loads(out_str)
    except json.JSONDecodeError as exc:
        print(f"    json parse failed: {exc}", file=sys.stderr)
        return None


# --------------------------- validation -------------------------------------

def validate_and_filter(record: dict) -> tuple[dict | None, str]:
    """
    Returns (cleaned_record, reason). cleaned_record is None on hard failure.
    Drops misaligned entities; keeps the CV if at least some entities survive.
    """
    if not isinstance(record, dict):
        return None, "not a dict"
    text = record.get("text") or ""
    if not isinstance(text, str) or len(text) < 200:
        return None, f"text too short ({len(text) if isinstance(text, str) else 0} chars)"

    title = record.get("title") or ""
    if not isinstance(title, str) or not title.strip():
        return None, "missing title"

    text_lower = text.lower()

    cleaned: dict = {"text": text, "title": title.strip()}
    miss_count = 0
    keep_count = 0

    for field in ("skills", "certifications", "projects"):
        items = record.get(field) or []
        if not isinstance(items, list):
            cleaned[field] = []
            continue
        kept: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            v = item.strip()
            if not v:
                continue
            if v.lower() in text_lower:
                kept.append(v)
                keep_count += 1
            else:
                miss_count += 1
        cleaned[field] = kept

    # Need at least one entity per field type ideally; but be lenient — accept
    # if at least 3 entities total survived alignment.
    if keep_count < 3:
        return None, f"too few aligned entities ({keep_count} kept, {miss_count} missed)"
    if miss_count > keep_count:
        return None, f"too many misalignments ({miss_count} missed, {keep_count} kept)"

    return cleaned, "ok"


# --------------------------- resumability -----------------------------------

def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("id") or ""))
            except json.JSONDecodeError:
                continue
    ids.discard("")
    return ids


# --------------------------- main loop --------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=500, help="Total CVs to generate")
    p.add_argument("--model", type=str, default="qwen2.5:7b")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries per CV when validation fails")
    p.add_argument("--show-samples", action="store_true",
                   help="Print first 3 generated CVs in full at the end")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Sanity-check Ollama
    err = check_ollama()
    if err:
        print(f"ERROR: Ollama unreachable at {OLLAMA_GENERATE_URL}", file=sys.stderr)
        print(f"  reason: {err}", file=sys.stderr)
        print(f"  fix:    run 'ollama serve' or check 'systemctl status ollama'", file=sys.stderr)
        sys.exit(1)

    completed = load_completed_ids(args.output)
    print(f"Model:     {args.model}")
    print(f"Output:    {args.output}")
    print(f"Existing:  {len(completed)} CVs")
    print(f"Target:    {args.count}")
    print()

    if len(completed) >= args.count:
        print(f"Already at target ({len(completed)} >= {args.count}). Nothing to do.")
        return

    rng = random.Random(args.seed)
    successes = 0
    hard_skips = 0
    t_start = time.perf_counter()

    with args.output.open("a", encoding="utf-8") as fout:
        i = 0
        while len(completed) + successes < args.count:
            i += 1
            cv_id = f"ollama-{len(completed) + successes + 1:04d}"

            # Diverse prompt parameters
            lang = rng.choices([l for l, _ in LANGUAGES], weights=[w for _, w in LANGUAGES])[0]
            industry = rng.choice(INDUSTRIES)
            level = rng.choice(LEVELS)

            prompt = PROMPT_TEMPLATE.format(
                language_full=LANG_FULL[lang],
                industry=industry,
                level=level,
            )

            # Try up to max_retries to get a valid record
            cleaned = None
            last_reason = ""
            for attempt in range(args.max_retries):
                raw = call_ollama(args.model, prompt)
                if raw is None:
                    last_reason = "ollama failed"
                    continue
                cleaned, last_reason = validate_and_filter(raw)
                if cleaned is not None:
                    break

            if cleaned is None:
                hard_skips += 1
                print(f"  [{cv_id}] SKIP after {args.max_retries} retries: {last_reason}",
                      file=sys.stderr)
                continue

            output_record = {
                "id": cv_id,
                "text": cleaned["text"],
                "labels": {
                    "title": cleaned["title"],
                    "skills": cleaned["skills"],
                    "certifications": cleaned["certifications"],
                    "projects": cleaned["projects"],
                },
                "meta": {
                    "source": "ollama_synthetic",
                    "model": args.model,
                    "language": lang,
                    "industry": industry,
                    "level": level,
                },
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()
            successes += 1

            done = len(completed) + successes
            if successes == 1 or successes % 10 == 0:
                elapsed = time.perf_counter() - t_start
                rate = successes / max(elapsed, 1.0)
                remaining = args.count - done
                eta_min = remaining / max(rate, 1e-3) / 60
                print(f"  [{done}/{args.count}] "
                      f"elapsed={elapsed/60:5.1f}m  "
                      f"rate={rate:.2f}cv/s  "
                      f"eta={eta_min:5.1f}m  "
                      f"skips={hard_skips}  "
                      f"({lang} / {industry[:20]} / {level[:15]})")

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"DONE — generated {successes} new records in {elapsed/60:.1f} min")
    print(f"Hard skips (validation failed all retries): {hard_skips}")
    print(f"Output: {args.output}")
    print(f"Total in file now: {len(completed) + successes}")

    if args.show_samples:
        print("\n=== Sample records ===")
        with args.output.open(encoding="utf-8") as f:
            for n, line in enumerate(f):
                if n >= 3:
                    break
                obj = json.loads(line)
                print(f"\n--- {obj.get('id')} [{obj.get('meta', {}).get('language')}] ---")
                print(f"title: {obj['labels'].get('title')}")
                print(f"skills: {obj['labels'].get('skills')}")
                print(f"certs:  {obj['labels'].get('certifications')}")
                print(f"projects: {obj['labels'].get('projects')}")
                print(f"text (first 400 chars): {obj['text'][:400]}...")


if __name__ == "__main__":
    main()
