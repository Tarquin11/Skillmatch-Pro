from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from app.services.cv_parser import extract_text


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _safe_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


_LANG_ALIAS = {
    "francais": "french",
    "français": "french",
    "french": "french",
    "anglais": "english",
    "english": "english",
    "arabe": "arabic",
    "arabic": "arabic",
    "espagnol": "spanish",
    "spanish": "spanish",
}


def _norm_lang(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return _LANG_ALIAS.get(raw, raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evaluate_cv_snippets-compatible gold JSONL from synthetic CV manifest.jsonl."
    )
    parser.add_argument("--manifest-jsonl", required=True, help="Path to artifacts/synth_cvs*/manifest.jsonl")
    parser.add_argument(
        "--pdf-dir",
        default="",
        help="Directory containing PDFs. Default: parent directory of manifest.",
    )
    parser.add_argument("--out-jsonl", required=True, help="Output gold JSONL for evaluate_cv_snippets")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of records")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_jsonl).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest JSONL not found: {manifest_path}")

    pdf_dir = Path(args.pdf_dir).expanduser() if args.pdf_dir else manifest_path.parent
    out_path = Path(args.out_jsonl).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(manifest_path)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    out_lines: list[str] = []
    skipped = 0
    for i, row in enumerate(rows, start=1):
        file_name = str(row.get("file", "")).strip()
        if not file_name:
            skipped += 1
            continue
        pdf_path = pdf_dir / file_name
        if not pdf_path.exists():
            skipped += 1
            continue

        text = extract_text(pdf_path.read_bytes(), pdf_path.name)
        labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}

        item = {
            "id": f"synth-{i:04d}",
            "text": text or "",
            "labels": {
                "skills": [str(x) for x in _safe_list(labels.get("skills")) if str(x).strip()],
                "tools": [str(x) for x in _safe_list(labels.get("tools")) if str(x).strip()],
                "languages": [
                    {"language": _norm_lang(d.get("language", "")), "level": d.get("level")}
                    for d in _safe_list(labels.get("languages"))
                    if isinstance(d, dict) and _norm_lang(d.get("language", ""))
                ],
                "title": None,
                "experience_years": None,
                "project_text": [],
            },
            "meta": {
                "source": "synthetic_manifest",
                "file": file_name,
                "very_dirty": bool(row.get("very_dirty")),
            },
        }
        out_lines.append(json.dumps(item, ensure_ascii=False))

    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Input rows: {len(rows)}")
    print(f"Written rows: {len(out_lines)}")
    print(f"Skipped rows: {skipped}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
