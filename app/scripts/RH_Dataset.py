import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.employee import Employee

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parents[1] / "Datasets"


@dataclass
class ImportStats:
    source_rows: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    invalid: int = 0
    conflicts: int = 0
    errors: int = 0


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _norm_emp_number(value: Any) -> str:
    return str(value).strip()


def _norm_email(value: Any) -> str:
    return str(value).strip().lower()


def parse_date(value: Any):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"N/A", "N/A - still employed"}:
        return None
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _first_last_name(full_name: str) -> tuple[str, str]:
    parts = (full_name or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[-1]


def _build_indexes(db: Session):
    by_number: dict[str, Employee] = {}
    by_email: dict[str, Employee] = {}

    for emp in db.query(Employee).all():
        num = _norm_emp_number(emp.employee_number)
        email = _norm_email(emp.email) if emp.email else ""
        if num:
            by_number[num] = emp
        if email:
            by_email[email] = emp

    return by_number, by_email


def _upsert_employee(
    db: Session,
    payload: dict[str, Any],
    stats: ImportStats,
    by_number: dict[str, Employee],
    by_email: dict[str, Employee],
):
    emp_number = _norm_emp_number(payload.get("employee_number", ""))
    email = _norm_email(payload.get("email", "")) if payload.get("email") else ""

    if not emp_number:
        stats.invalid += 1
        return

    payload["employee_number"] = emp_number
    if email:
        payload["email"] = email

    by_num_emp = by_number.get(emp_number)
    by_email_emp = by_email.get(email) if email else None

    # conflicting keys point to different rows
    if by_num_emp and by_email_emp and by_num_emp is not by_email_emp:
        stats.conflicts += 1
        logger.warning(
            "Conflict: employee_number=%s and email=%s map to different employees; skipping row.",
            emp_number,
            email,
        )
        return

    existing = by_num_emp or by_email_emp

    if not existing:
        new_emp = Employee(**payload)
        db.add(new_emp)

        # keep in-memory indexes in sync for this same run
        by_number[emp_number] = new_emp
        if email:
            by_email[email] = new_emp

        stats.created += 1
        return

    changed = False
    for field, new_value in payload.items():
        if new_value is None:
            continue
        if isinstance(new_value, str) and not new_value.strip():
            continue

        if field == "email":
            new_value = _norm_email(new_value)
            old_email = _norm_email(existing.email) if existing.email else ""

            if new_value != old_email:
                conflict_emp = by_email.get(new_value)
                if conflict_emp and conflict_emp is not existing:
                    stats.conflicts += 1
                    logger.warning(
                        "Conflict: email=%s already used by another employee; skipping email update.",
                        new_value,
                    )
                    continue

                if old_email and by_email.get(old_email) is existing:
                    del by_email[old_email]
                by_email[new_value] = existing

        old_value = getattr(existing, field, None)
        if old_value != new_value:
            setattr(existing, field, new_value)
            changed = True

    by_number[emp_number] = existing

    if changed:
        stats.updated += 1
    else:
        stats.unchanged += 1


def _import_hrdataset_v9(
    db: Session,
    stats: ImportStats,
    by_number: dict[str, Employee],
    by_email: dict[str, Employee],
):
    file_path = DATA_DIR / "HRDataset_v9.csv"
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        stats.source_rows += 1
        try:
            emp_number = _norm_emp_number(row.get("Employee Number", ""))
            if not emp_number:
                stats.invalid += 1
                continue

            full_name = _clean_text(row.get("Employee Name")) or ""
            first_name, last_name = _first_last_name(full_name)
            email = f"{emp_number}@company.local"

            payload = {
                "employee_number": emp_number,
                "full_name": full_name or f"{first_name} {last_name}".strip(),
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "gender": _clean_text(row.get("Sex")),
                "dob": parse_date(row.get("DOB")),
                "department": _clean_text(row.get("Department")),
                "position": _clean_text(row.get("Position")),
                "pay_rate": row.get("Pay Rate"),
                "performance_score": _clean_text(row.get("Performance Score")),
                "hire_date": parse_date(row.get("Date of Hire")),
                "termination_date": parse_date(row.get("Date of Termination")),
                "employment_status": _clean_text(row.get("Employment Status")) or "active",
            }

            _upsert_employee(db, payload, stats, by_number, by_email)
        except Exception:
            stats.errors += 1
            logger.exception("Failed row in HRDataset_v9.csv")


def _import_humanresources(
    db: Session,
    stats: ImportStats,
    by_number: dict[str, Employee],
    by_email: dict[str, Employee],
):
    file_path = DATA_DIR / "HumanResources.csv"
    df = pd.read_csv(file_path, sep=";")

    for _, row in df.iterrows():
        stats.source_rows += 1
        try:
            emp_number = _norm_emp_number(row.get("Employee_ID", ""))
            if not emp_number:
                stats.invalid += 1
                continue

            first_name = _clean_text(row.get("First Name")) or ""
            last_name = _clean_text(row.get("Last Name")) or ""
            full_name = f"{first_name} {last_name}".strip()
            email = _clean_text(row.get("Email")) or f"{emp_number}@company.local"

            payload = {
                "employee_number": emp_number,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name or f"Employee {emp_number}",
                "email": email,
                "gender": _clean_text(row.get("Gender")),
                "dob": parse_date(row.get("Birthdate")),
                "department": _clean_text(row.get("Department")),
                "position": _clean_text(row.get("Job Title")),
                "salary": row.get("Salary"),
                "performance_score": _clean_text(row.get("Performance Rating")),
                "hire_date": parse_date(row.get("Hiredate")),
                "termination_date": parse_date(row.get("Termdate")),
            }

            _upsert_employee(db, payload, stats, by_number, by_email)
        except Exception:
            stats.errors += 1
            logger.exception("Failed row in HumanResources.csv")


def import_hr_data() -> ImportStats:
    stats = ImportStats()
    db = SessionLocal()
    try:
        by_number, by_email = _build_indexes(db)

        _import_hrdataset_v9(db, stats, by_number, by_email)
        _import_humanresources(db, stats, by_number, by_email)

        db.commit()

        logger.info(
            "HR import summary | rows=%d created=%d updated=%d unchanged=%d invalid=%d conflicts=%d errors=%d",
            stats.source_rows,
            stats.created,
            stats.updated,
            stats.unchanged,
            stats.invalid,
            stats.conflicts,
            stats.errors,
        )
        return stats
    except Exception:
        db.rollback()
        logger.exception("HR import failed; transaction rolled back.")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    import_hr_data()
