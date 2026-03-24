import os
from dotenv import load_dotenv

load_dotenv()

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


class Settings:
    PROJECT_NAME: str = "SkillMatch Pro"
    SECRET_KEY: str = _get_required_env("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    LOGIN_RATE_LIMIT_MAX_ATTEMPTS: int = int(os.getenv("LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "10"))
    LOGIN_RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "60"))
    LOGIN_MAX_FAILED_ATTEMPTS: int = int(os.getenv("LOGIN_MAX_FAILED_ATTEMPTS", "5"))
    LOGIN_LOCKOUT_BASE_SECONDS: int = int(os.getenv("LOGIN_LOCKOUT_BASE_SECONDS", "30"))
    LOGIN_LOCKOUT_MAX_SECONDS: int = int(os.getenv("LOGIN_LOCKOUT_MAX_SECONDS", "900"))
    DATABASE_URL: str = _get_required_env("DATABASE_URL")
    AI_MODEL_PATH: str = os.getenv("AI_MODEL_PATH", "artifacts/matcher.joblib")
    AI_MODEL_AUTOLOAD: bool = os.getenv("AI_MODEL_AUTOLOAD", "true").strip().lower() in {"1", "true", "yes", "on"}
    AI_MODEL_VERSION: str = os.getenv("AI_MODEL_VERSION", "dev")

settings = Settings()
