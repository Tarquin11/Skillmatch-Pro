import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment from explicit path to ensure it's found when running as a module
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Set HF token if available
hf_token = os.getenv("HF_TOKEN", "").strip()
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")

def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

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
    AI_MODEL_AUTOLOAD: bool = _get_bool_env("AI_MODEL_AUTOLOAD", True)
    AI_MODEL_VERSION: str = os.getenv("AI_MODEL_VERSION", "dev")
    AI_CANARY_ENABLED: bool = _get_bool_env("AI_CANARY_ENABLED", False)
    AI_CANARY_MODEL_PATH: str = os.getenv("AI_CANARY_MODEL_PATH", "artifacts/matcher_canary.joblib")
    AI_CANARY_TRAFFIC_PERCENT: float = max(0.0, min(100.0, _get_float_env("AI_CANARY_TRAFFIC_PERCENT", 0.0)))

    CV_PARSER_MODEL_VERSION: str = os.getenv("CV_PARSER_MODEL_VERSION", "cv_parser_v1")
    CV_PARSER_MIN_CONFIDENCE: float = max(0.0, min(1.0, _get_float_env("CV_PARSER_MIN_CONFIDENCE", 0.6)))
    CV_PARSER_USE_SEMANTIC: bool = _get_bool_env("CV_PARSER_USE_SEMANTIC", False)
    CV_PARSER_USE_HF_NER: bool = _get_bool_env("CV_PARSER_USE_HF_NER", True)
    CV_PARSER_USE_SEMANTIC_AUGMENT: bool = _get_bool_env("CV_PARSER_USE_SEMANTIC_AUGMENT", True)
    CV_PARSER_SKILL_TIME_BUDGET_SECONDS: float = max(0.05, _get_float_env("CV_PARSER_SKILL_TIME_BUDGET_SECONDS", 0.75))
    CV_PARSER_SLO_MS: int = max(200, _get_int_env("CV_PARSER_SLO_MS", 2000))

    

settings = Settings()
