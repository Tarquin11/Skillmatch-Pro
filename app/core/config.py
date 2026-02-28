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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str = _get_required_env("DATABASE_URL")


settings = Settings()