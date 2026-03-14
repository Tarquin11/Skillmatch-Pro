from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

NON_ALNUM_RE = re.compile(r"[^a-z0-9+.#/\-\s]")
def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("&", " and ")
    text = NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def _load_esco_occupations(path: Path) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    label_index: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"ESCO occupations not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            occ_id = str(row.get("id", "")).strip()
            label = str(row.get("preferred_label", "")).strip()
            if occ_id:
                ids.add(occ_id)
            if label:
                label_index[_normalize_key(label)] = occ_id
            for alt in row.get("alt_labels", []) or []:
                alt_key = _normalize_key(alt)
                if alt_key and alt_key not in label_index:
                    label_index[alt_key] = occ_id
    return ids, label_index

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate O*NET -> ESCO mapping CSV")
    parser.add_argument("--mapping", default="data/taxonomy/onet_mapping.csv")
    parser.add_argument("--esco-occupations", default="data/taxonomy/esco_occupations.jsonl")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_path}")
    esco_ids, label_index = _load_esco_occupations(Path(args.esco_occupations))
    errors: list[str] = []
    warnings: list[str] = []
    seen_onet: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"onet_code", "onet_title", "esco_id", "esco_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns: {missing}")
        for idx, row in enumerate(reader, start=2):
            onet_code = (row.get("onet_code") or "").strip()
            onet_title = (row.get("onet_title") or "").strip()
            esco_id = (row.get("esco_id") or "").strip()
            esco_label = (row.get("esco_label") or "").strip()

            if not onet_code and not onet_title:
                warnings.append(f"Row {idx}: missing O*NET code and title.")

            if onet_code:
                prev = seen_onet.get(onet_code)
                if prev and prev != esco_id:
                    warnings.append(
                        f"Row {idx}: O*NET code {onet_code} mapped to multiple ESCO IDs."
                    )
                if esco_id:
                    seen_onet[onet_code] = esco_id

            if not esco_id and not esco_label:
                warnings.append(f"Row {idx}: missing ESCO id and label.")
                continue

            if esco_id and esco_id not in esco_ids:
                errors.append(f"Row {idx}: ESCO id not found: {esco_id}")

            if esco_label:
                key = _normalize_key(esco_label)
                if key not in label_index:
                    warnings.append(f"Row {idx}: ESCO label not found: {esco_label}")
                elif esco_id and label_index[key] != esco_id:
                    warnings.append(
                        f"Row {idx}: ESCO id/label mismatch: {esco_id} vs {esco_label}"
                    )

    print(f"Validation complete. Errors: {len(errors)} Warnings: {len(warnings)}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"- {e}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")

    if errors or (warnings and args.fail_on_warning):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
