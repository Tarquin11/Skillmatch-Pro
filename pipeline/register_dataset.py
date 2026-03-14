from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _time_now() -> str :
    return datetime.now(timezone.utc).isoformat()

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _count_records(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix in { ".jsonl", "csv"}:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            count = sum(1 for _ in f)
        return max(0, count - 1) if suffix == ".csv" else count 
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict) and "pairs" in data and isinstance(data["pairs"], list):
            return len(data["pairs"])
        if isinstance(data, list):
            return len(data)
        return None
    
def _load_registery(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try : 
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []
    
def _save_registery(path: Path , registery: list [dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registery, indent=2), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Register dataset with license metadata")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--license", default="")
    parser.add_argument("--license-url", default="")
    parser.add_argument("--usage-rights", default="")
    parser.add_argument("--pii", default="unknown")
    parser.add_argument("--schema-version", default="pairs_v1")
    parser.add_argument("--notes", default="")
    parser.add_argument("--registry", default="data/registry/datasets_registry.json")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    record_count = _count_records(raw_path)
    entry = {
        "dataset_id": args.dataset_id,
        "source": args.source,
        "origin": args.origin,
        "license": args.license,
        "license_url": args.license_url,
        "usage_rights": args.usage_rights,
        "pii": args.pii,
        "terms_accepted_by": "",
        "terms_accepted_at_utc": "",
        "ingested_at_utc": _time_now(),
        "raw_path": str(raw_path),
        "record_count": record_count if record_count is not None else 0,
        "sha256": _sha256(raw_path),
        "schema_version": args.schema_version,
        "notes": args.notes,
    }

    registry_path = Path(args.registry)
    registry = _load_registery(registry_path)
    registry = [r for r in registry if r.get("dataset_id") != args.dataset_id]
    registry.append(entry)
    _save_registery(registry_path, registry)

    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()