"""
One-shot user-table cleanup.

Keeps exactly three users with the requested roles:
    zarrouk.moataz2003@gmail.com  → admin
    user@skillmatch.local         → user
    admin@skillmatch.local        → recruiter

All other users (and their refresh tokens via cascade) are deleted. Audit
fields (`created_by`) on records owned by deleted users are set to NULL so
foreign-key constraints don't block the delete.

Usage:
    venv/bin/python -m app.scripts.reset_users [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import inspect, text

from app.db.database import SessionLocal, engine
from app.models.user import User

KEEP = {
    "zarrouk.moataz2003@gmail.com": "admin",
    "user@skillmatch.local":        "user",
    "admin@skillmatch.local":       "recruiter",
}


def _null_out_created_by(db, doomed_user_ids: list[int]) -> None:
    """Find every table with a `created_by` column referencing users.id and
    NULL out rows where created_by is in doomed_user_ids. Otherwise the
    delete will fail on the FK constraint."""
    if not doomed_user_ids:
        return
    insp = inspect(engine)
    placeholder = ",".join(str(int(i)) for i in doomed_user_ids)
    for table_name in insp.get_table_names():
        if table_name == "users":
            continue
        cols = {c["name"] for c in insp.get_columns(table_name)}
        if "created_by" not in cols:
            continue
        result = db.execute(
            text(f"UPDATE {table_name} SET created_by = NULL "
                 f"WHERE created_by IN ({placeholder})")
        )
        if result.rowcount:
            print(f"    null-out: {table_name}.created_by × {result.rowcount}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan, don't commit")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        all_users = db.query(User).order_by(User.id).all()
        print(f"Current users in DB: {len(all_users)}")
        for u in all_users:
            mark = "KEEP" if u.email in KEEP else "DELETE"
            print(f"  [{mark:>6}] id={u.id:<3} email={u.email!r:<45} role={u.role}")

        # 1. Update kept users' roles
        kept_count = 0
        for email, role in KEEP.items():
            user = db.query(User).filter(User.email == email).first()
            if user is None:
                print(f"  WARN: kept user {email!r} does not exist in DB — skipped (create it manually)")
                continue
            if user.role != role:
                print(f"  set role {user.email!r}: {user.role!r} → {role!r}")
                user.role = role
                user.token_version = (user.token_version or 0) + 1  # invalidate sessions
            kept_count += 1

        # 2. Identify users to delete
        doomed = [u for u in all_users if u.email not in KEEP]
        doomed_ids = [int(u.id) for u in doomed]

        if doomed:
            print(f"\n  null-out audit FK on {len(doomed)} doomed users…")
            _null_out_created_by(db, doomed_ids)
            for u in doomed:
                print(f"  delete: id={u.id} email={u.email!r}")
                db.delete(u)

        if args.dry_run:
            print("\n--dry-run: rollback")
            db.rollback()
        else:
            db.commit()
            print("\nDone.")
            print(f"  kept:    {kept_count} / {len(KEEP)} requested")
            print(f"  deleted: {len(doomed)}")
    except Exception as exc:
        db.rollback()
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
