from __future__ import annotations
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator

ALT_LABEL_SPLIT_RE = re.compile(r"[;\r\n]+")
NON_ALNUM_RE = re.compile(r"[a-z0-9+.#/\-\s]")

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\u00a0\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ",text)
    return text.strip()

def _normalize_key(value: Any) -> str:
    text = _normalize_text(value).replace("&", "and")
    text = NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def _split_alt_labels(value: Any) -> list[str]:
    if not value: 
        return []
    parts = ALT_LABEL_SPLIT_RE.split(str(value))
    return [p.strip() for p in parts if p.strip()]

def _read_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v or "") for k , v in row.items()}

def _write_jsonl(rows: Iterable[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

def load_occupations(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "occupations.csv"
    occupations: list[dict[str, Any]] = []
    for row in _read_csv(path):
        preferred = row.get("PREFERREDLABEL", "").strip()
        alt = _split_alt_labels(row.get("ALTLABELS", ""))
        labels = [preferred] + alt if preferred else alt
        occupations.append(
            {
                "id": row.get("ID", ""),
                "origin_uri": row.get("ORIGINURI", ""),
                "isco_group_code": row.get("ISCOGROUPCODE", ""),
                "code": row.get("CODE", ""),
                "preferred_label": preferred,
                "alt_labels": alt,
                "labels": labels,
                "description": row.get("DESCRIPTION", ""),
                "definition": row.get("DEFINITION", ""),
                "scope_note": row.get("SCOPENOTE", ""),
                "occupation_type": row.get("OCCUPATIONTYPE", ""),
                "is_localized": row.get("ISLOCALIZED", ""),
            }
        )
    return occupations


def load_skills(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "skills.csv"
    skills: list[dict[str, Any]] = []
    for row in _read_csv(path):
        preferred = row.get("PREFERREDLABEL", "").strip()
        alt = _split_alt_labels(row.get("ALTLABELS", ""))
        labels = [preferred] + alt if preferred else alt
        skills.append(
            {
                "id": row.get("ID", ""),
                "origin_uri": row.get("ORIGINURI", ""),
                "skill_type": row.get("SKILLTYPE", ""),
                "reuse_level": row.get("REUSELEVEL", ""),
                "preferred_label": preferred,
                "alt_labels": alt,
                "labels": labels,
                "description": row.get("DESCRIPTION", ""),
                "definition": row.get("DEFINITION", ""),
                "scope_note": row.get("SCOPENOTE", ""),
            }
        )
    return skills


def load_occupation_skill_relations(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "occupation_skill_relations.csv"
    rels: list[dict[str, Any]] = []
    for row in _read_csv(path):
        rels.append(
            {
                "occupation_id": row.get("OCCUPATIONID", ""),
                "relation_type": row.get("RELATIONTYPE", ""),
                "skill_id": row.get("SKILLID", ""),
                "occupation_type": row.get("OCCUPATIONTYPE", ""),
            }
        )
    return rels


def load_hierarchy(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "occupations_hierarchy.csv"
    edges: list[dict[str, Any]] = []
    for row in _read_csv(path):
        edges.append(
            {
                "parent_type": row.get("PARENTOBJECTTYPE", ""),
                "parent_id": row.get("PARENTID", ""),
                "child_id": row.get("CHILDID", ""),
                "child_type": row.get("CHILDOBJECTTYPE", ""),
            }
        )
    return edges


def load_skill_hierarchy(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "skills_hierarchy.csv"
    edges: list[dict[str, Any]] = []
    for row in _read_csv(path):
        edges.append(
            {
                "parent_type": row.get("PARENTOBJECTTYPE", ""),
                "parent_id": row.get("PARENTID", ""),
                "child_id": row.get("CHILDID", ""),
                "child_type": row.get("CHILDOBJECTTYPE", ""),
            }
        )
    return edges


def load_skill_relations(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "skill_skill_relations.csv"
    rels: list[dict[str, Any]] = []
    for row in _read_csv(path):
        rels.append(
            {
                "requiring_id": row.get("REQUIRINGID", ""),
                "relation_type": row.get("RELATIONTYPE", ""),
                "required_id": row.get("REQUIREDID", ""),
            }
        )
    return rels


def load_isco_groups(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "ISCOGroups.csv"
    groups: list[dict[str, Any]] = []
    for row in _read_csv(path):
        groups.append(
            {
                "id": row.get("ID", ""),
                "origin_uri": row.get("ORIGINURI", ""),
                "code": row.get("CODE", ""),
                "preferred_label": row.get("PREFERREDLABEL", ""),
                "alt_labels": _split_alt_labels(row.get("ALTLABELS", "")),
                "description": row.get("DESCRIPTION", ""),
            }
        )
    return groups


def load_skill_groups(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "skillGroups.csv"
    groups: list[dict[str, Any]] = []
    for row in _read_csv(path):
        groups.append(
            {
                "id": row.get("ID", ""),
                "origin_uri": row.get("ORIGINURI", ""),
                "code": row.get("CODE", ""),
                "preferred_label": row.get("PREFERREDLABEL", ""),
                "alt_labels": _split_alt_labels(row.get("ALTLABELS", "")),
                "description": row.get("DESCRIPTION", ""),
                "scope_note": row.get("SCOPENOTE", ""),
            }
        )
    return groups


def _build_label_index(occupations: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for occ in occupations:
        occ_id = occ.get("id", "")
        for label in occ.get("labels", []):
            key = _normalize_key(label)
            if not key:
                continue
            index.setdefault(key, []).append({"id": occ_id, "label": label})
    return index


def _label_lookup(occupations: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for occ in occupations:
        occ_id = occ.get("id", "")
        for label in occ.get("labels", []):
            key = _normalize_key(label)
            if key:
                rows.append((occ_id, label, key))
    return rows


def _read_internal_titles(path: Path) -> list[dict[str, Any]]:
    titles: list[dict[str, Any]] = []
    if not path.exists():
        return titles
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("internal_title") or "").strip()
            aliases_raw = row.get("aliases") or ""
            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            if title:
                titles.append({"internal_title": title, "aliases": aliases})
    return titles


def _match_title(
    title: str,
    label_index: dict[str, list[dict[str, str]]],
    label_lookup: list[tuple[str, str, str]],
    fuzzy_threshold: float,
) -> dict[str, Any]:
    key = _normalize_key(title)
    if not key:
        return {"match_type": "empty", "confidence": 0.0}

    exact_candidates = label_index.get(key, [])
    if len(exact_candidates) == 1:
        cand = exact_candidates[0]
        return {
            "match_type": "exact",
            "confidence": 1.0,
            "matched_esco_id": cand["id"],
            "matched_esco_label": cand["label"],
        }
    if len(exact_candidates) > 1:
        return {
            "match_type": "ambiguous",
            "confidence": 1.0,
            "candidate_esco_ids": [c["id"] for c in exact_candidates],
        }

    best_ratio = 0.0
    best = None
    for occ_id, label, norm in label_lookup:
        ratio = SequenceMatcher(None, key, norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = (occ_id, label)
    if best and best_ratio >= fuzzy_threshold:
        return {
            "match_type": "fuzzy",
            "confidence": round(best_ratio, 4),
            "matched_esco_id": best[0],
            "matched_esco_label": best[1],
        }

    return {"match_type": "unmatched", "confidence": 0.0}


def build_internal_mapping(
    internal_titles: list[dict[str, Any]],
    label_index: dict[str, list[dict[str, str]]],
    label_lookup: list[tuple[str, str, str]],
    fuzzy_threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in internal_titles:
        title = row["internal_title"]
        aliases = row.get("aliases", [])
        match = _match_title(title, label_index, label_lookup, fuzzy_threshold)
        out.append(
            {
                "internal_title": title,
                "normalized_title": _normalize_key(title),
                "aliases": aliases,
                **match,
            }
        )
    return out


def _read_onet_mapping(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def apply_onet_mapping(
    occupations: list[dict[str, Any]],
    label_index: dict[str, list[dict[str, str]]],
    onet_mapping: list[dict[str, str]],
) -> None:
    occ_by_id = {o.get("id"): o for o in occupations}
    for row in onet_mapping:
        onet_code = row.get("onet_code", "").strip()
        onet_title = row.get("onet_title", "").strip()
        esco_id = row.get("esco_id", "").strip()
        esco_label = row.get("esco_label", "").strip()

        target = None
        if esco_id and esco_id in occ_by_id:
            target = occ_by_id[esco_id]
        elif esco_label:
            key = _normalize_key(esco_label)
            candidates = label_index.get(key, [])
            if len(candidates) == 1:
                target = occ_by_id.get(candidates[0]["id"])

        if target:
            target["onet"] = {"code": onet_code, "title": onet_title}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build job taxonomy from Tabiya ESCO dataset")
    parser.add_argument("--dataset-dir", default="tabiya_dataset")
    parser.add_argument("--out-dir", default="data/taxonomy")
    parser.add_argument("--internal-titles", default="data/taxonomy/internal_titles.csv")
    parser.add_argument("--onet-mapping", default="")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.92)
    parser.add_argument("--version", default="")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    occupations = load_occupations(dataset_dir)
    skills = load_skills(dataset_dir)
    occ_skill = load_occupation_skill_relations(dataset_dir)
    occ_hierarchy = load_hierarchy(dataset_dir)
    skill_hierarchy = load_skill_hierarchy(dataset_dir)
    skill_relations = load_skill_relations(dataset_dir)
    isco_groups = load_isco_groups(dataset_dir)
    skill_groups = load_skill_groups(dataset_dir)

    label_index = _build_label_index(occupations)
    label_lookup = _label_lookup(occupations)

    internal_titles = _read_internal_titles(Path(args.internal_titles))
    internal_mapping = build_internal_mapping(
        internal_titles, label_index, label_lookup, args.fuzzy_threshold
    )

    if args.onet_mapping:
        onet_rows = _read_onet_mapping(Path(args.onet_mapping))
        apply_onet_mapping(occupations, label_index, onet_rows)

    _write_jsonl(occupations, out_dir / "esco_occupations.jsonl")
    _write_jsonl(skills, out_dir / "esco_skills.jsonl")
    _write_jsonl(occ_skill, out_dir / "esco_occupation_skill.jsonl")
    _write_jsonl(occ_hierarchy, out_dir / "esco_occupations_hierarchy.jsonl")
    _write_jsonl(skill_hierarchy, out_dir / "esco_skills_hierarchy.jsonl")
    _write_jsonl(skill_relations, out_dir / "esco_skill_relations.jsonl")
    _write_jsonl(isco_groups, out_dir / "isco_groups.jsonl")
    _write_jsonl(skill_groups, out_dir / "skill_groups.jsonl")
    if internal_mapping:
        _write_jsonl(internal_mapping, out_dir / "internal_title_mapping.jsonl")

    taxonomy_summary = {
        "taxonomy_version": args.version or _now_utc(),
        "generated_at_utc": _now_utc(),
        "source": {"type": "ESCO", "dataset_dir": str(dataset_dir)},
        "counts": {
            "occupations": len(occupations),
            "skills": len(skills),
            "occupation_skill_relations": len(occ_skill),
            "occupation_hierarchy_edges": len(occ_hierarchy),
            "skill_hierarchy_edges": len(skill_hierarchy),
            "skill_skill_relations": len(skill_relations),
            "isco_groups": len(isco_groups),
            "skill_groups": len(skill_groups),
            "internal_title_mappings": len(internal_mapping),
        },
        "outputs": {
            "occupations": str(out_dir / "esco_occupations.jsonl"),
            "skills": str(out_dir / "esco_skills.jsonl"),
            "occupation_skill_relations": str(out_dir / "esco_occupation_skill.jsonl"),
            "occupation_hierarchy": str(out_dir / "esco_occupations_hierarchy.jsonl"),
            "skill_hierarchy": str(out_dir / "esco_skills_hierarchy.jsonl"),
            "skill_skill_relations": str(out_dir / "esco_skill_relations.jsonl"),
            "isco_groups": str(out_dir / "isco_groups.jsonl"),
            "skill_groups": str(out_dir / "skill_groups.jsonl"),
            "internal_title_mapping": str(out_dir / "internal_title_mapping.jsonl"),
        },
    }

    summary_path = out_dir / "job_taxonomy.json"
    summary_path.write_text(json.dumps(taxonomy_summary, indent=2), encoding="utf-8")
    print(f"Taxonomy built: {summary_path}")


if __name__ == "__main__":
    main()