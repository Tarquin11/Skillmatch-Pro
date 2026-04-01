import re

from app.ai.preprocessing import normalize_skill_name


_ALIAS_MAP: dict[str, str] = {
    "my sql": "mysql",
    "mysql": "mysql",
    "ms sql": "mssql",
    "ms-sql": "mssql",
    "mssql": "mssql",
    "microsoft sql server": "mssql",
    "azure web services": "aws",
    "amazon web services": "aws",
    "java j2ee": "java/j2ee",
    "java/j2ee": "java/j2ee",
    "c plus plus": "c++",
    "c sharp": "c#",
}


def canonicalize_skill(raw: str | None) -> str:
    normalized = normalize_skill_name(raw or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    return _ALIAS_MAP.get(normalized, normalized)
