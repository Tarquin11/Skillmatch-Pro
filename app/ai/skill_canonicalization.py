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
    "burpsuite": "burp suite",
    "burp suite community": "burp suite",
    "js": "javascript",
    "javascript es6": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "ts": "typescript",
    "postgres": "postgresql",
    "postgre": "postgresql",
    "postgre sql": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "docker compose": "docker",
    "reactjs": "react",
    "vuejs": "vue",
    "nuxtjs": "nuxt",
    "nextjs": "next.js",
    "html5": "html",
    "html 5": "html",
    "zend framework": "zend",
    "symfony framework": "symfony",
    "code igniter": "codeigniter",
    "rest api": "rest",
    "restful api": "rest",
    "machine learning": "ml",
    "deep learning": "dl",
}


def canonicalize_skill(raw: str | None) -> str:
    normalized = normalize_skill_name(raw or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    return _ALIAS_MAP.get(normalized, normalized)
