from fastapi import FastAPI
from app.api import auth
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.ai.runtime import load_matcher_artifact
from app.core.config import settings
from app.api.employees import router as employees_router
from app.api.candidates import router as candidates_router
from app.api.match import router as match_router
from app.api.jobs import router as jobs_router
from app.api.skills import router as skills_router
from app.api.department import router as departements_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.AI_MODEL_AUTOLOAD:
        matcher = load_matcher_artifact(settings.AI_MODEL_PATH)
        if matcher is None:
            logger.warning(
                "No AI model artifact found at %s (fallback scoring will be used).",
                settings.AI_MODEL_PATH,
            )
        else:
            logger.info("Loaded AI model artifact from %s", settings.AI_MODEL_PATH)
    else:
        logger.info("AI model autoload disabled.")
    yield


app = FastAPI(title="SkillMatch Pro", lifespan=lifespan)
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(employees_router, prefix="/employees", tags=["Employees"])
app.include_router(candidates_router)
app.include_router(match_router)
app.include_router(jobs_router)
app.include_router(skills_router)
app.include_router(departements_router)

@app.get("/")
def health_check():
    return {"status": "ok"}