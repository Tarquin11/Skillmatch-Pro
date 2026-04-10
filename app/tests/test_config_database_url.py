from pathlib import Path

from app.core.config import _resolve_database_url


def test_resolve_database_url_converts_relative_sqlite_path():
    root = Path("C:/workspace/project")
    got = _resolve_database_url("sqlite:///./skillmatch.db", root)
    assert got.startswith("sqlite:///")
    assert got.endswith("/workspace/project/skillmatch.db")


def test_resolve_database_url_keeps_absolute_sqlite_path():
    root = Path("/workspace/project")
    url = "sqlite:////var/data/skillmatch.db"
    assert _resolve_database_url(url, root) == url


def test_resolve_database_url_keeps_non_sqlite_url():
    root = Path("/workspace/project")
    url = "postgresql://user:pw@localhost:5432/skillmatch"
    assert _resolve_database_url(url, root) == url
