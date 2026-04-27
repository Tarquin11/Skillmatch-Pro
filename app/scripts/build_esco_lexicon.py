"""
Build a multilingual lexicon from ESCO CSVs for NER weak supervision.

Reads the official ESCO French and English language packs (preferredLabel +
altLabels per concept) and emits a single unified JSON lexicon mapping
normalized surface forms to their entity type, ESCO URI, language, and
canonical label.

The output feeds two downstream uses:
  1. Lexicon-based weak supervision (gazetteer tagging) over CV text to
     auto-generate BIO-tagged training data for NER fine-tuning.
  2. Runtime canonicalization: when the NER model emits a span, look it up
     in the lexicon to recover the official ESCO concept.

Usage:
    python -m app.scripts.build_esco_lexicon
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "lexicons"
OUTPUT_FILE = OUTPUT_DIR / "esco_lexicon.json"

# Language packs to process. Order matters: when the same surface form
# appears in multiple languages, the FIRST pack wins. French is processed
# first so French-only terms get French priority on canonicalization.
PACKS: list[tuple[str, Path]] = [
    ("fr", ROOT / "app" / "ESCO_French_Dataset"),
    ("en", ROOT / "app" / "ESCO_English_Dataset"),
]

# Source files in priority order within each pack. When the same surface form
# appears in multiple files of the SAME pack, the FIRST file wins — so put the
# most specific subtypes before the general skills_{lang}.csv.
SOURCES: list[tuple[str, str, str | None]] = [
    # (filename_template, entity_type, subtype)
    ("transversalSkillsCollection_{lang}.csv", "SOFT_SKILL", None),
    ("languageSkillsCollection_{lang}.csv",    "LANGUAGE",   None),
    ("digitalSkillsCollection_{lang}.csv",     "SKILL",      "digital"),
    ("skills_{lang}.csv",                      "SKILL",      "general"),
    ("occupations_{lang}.csv",                 "TITLE",      None),
]

MIN_LABEL_LEN = 3
MAX_LABEL_LEN = 80


_PAREN_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_alt_labels(raw: str):
    """ESCO altLabels are either newline-separated OR pipe-separated. Handle both."""
    if not raw:
        return
    for chunk in raw.split("\n"):
        for label in re.split(r"\s*\|\s*", chunk):
            label = label.strip()
            if label:
                yield label


def expand_label_variants(label: str):
    """
    Yield the label and any useful variants — currently the version with the
    trailing parenthetical qualifier stripped. ESCO uses qualifiers to
    disambiguate (e.g. "Python (programmation informatique)") but real CV text
    rarely includes them.
    """
    yield label
    stripped = _PAREN_QUALIFIER_RE.sub("", label).strip()
    if stripped and stripped != label:
        yield stripped


def is_acceptable_label(label: str) -> bool:
    if not label:
        return False
    n = len(label)
    if n < MIN_LABEL_LEN or n > MAX_LABEL_LEN:
        return False
    if label.isdigit():
        return False
    return True


def process_pack(pack_lang: str, pack_dir: Path, lexicon: dict, stats: dict):
    if not pack_dir.exists():
        print(f"  WARN: pack dir not found: {pack_dir}", file=sys.stderr)
        return

    print(f"\n[{pack_lang}] {pack_dir}")
    pack_stats: dict[str, dict] = {}

    for filename_tpl, entity_type, subtype in SOURCES:
        filename = filename_tpl.format(lang=pack_lang)
        path = pack_dir / filename
        if not path.exists():
            print(f"  WARN: {filename} not found, skipping", file=sys.stderr)
            continue

        added = 0
        skipped_dupe = 0
        skipped_filter = 0

        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = (row.get("conceptUri") or "").strip()
                preferred = (row.get("preferredLabel") or "").strip()
                alt_labels = list(split_alt_labels(row.get("altLabels") or ""))

                # preferred first so it wins on intra-pack dedup
                candidates = [(preferred, False)] + [(a, True) for a in alt_labels]

                for raw_label, is_alt in candidates:
                    for label in expand_label_variants(raw_label):
                        if not is_acceptable_label(label):
                            skipped_filter += 1
                            continue
                        key = normalize_for_match(label)
                        if key in lexicon:
                            skipped_dupe += 1
                            continue
                        lexicon[key] = {
                            "preferred_label": preferred,
                            "surface_form": label,
                            "entity_type": entity_type,
                            "subtype": subtype,
                            "esco_uri": uri,
                            "is_alt_label": is_alt,
                            "language": pack_lang,
                        }
                        added += 1

        pack_stats[filename] = {
            "added": added,
            "skipped_duplicate": skipped_dupe,
            "skipped_filter": skipped_filter,
        }
        print(f"  {filename}: +{added} (dupes: {skipped_dupe}, filtered: {skipped_filter})")

    stats[pack_lang] = pack_stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lexicon: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    for pack_lang, pack_dir in PACKS:
        process_pack(pack_lang, pack_dir, lexicon, stats)

    by_type: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for entry in lexicon.values():
        by_type[entry["entity_type"]] = by_type.get(entry["entity_type"], 0) + 1
        by_lang[entry["language"]] = by_lang.get(entry["language"], 0) + 1

    output = {
        "metadata": {
            "source": "ESCO multilingual classification (FR + EN)",
            "total_entries": len(lexicon),
            "by_entity_type": by_type,
            "by_language": by_lang,
            "by_pack": stats,
            "filters": {
                "min_label_len": MIN_LABEL_LEN,
                "max_label_len": MAX_LABEL_LEN,
            },
        },
        "entries": lexicon,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 50}")
    print(f"Total entries: {len(lexicon)}")
    print("By entity type:")
    for t, n in sorted(by_type.items()):
        print(f"  {t:<12} {n}")
    print("By language:")
    for l, n in sorted(by_lang.items()):
        print(f"  {l:<12} {n}")
    print(f"\nWritten to: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
