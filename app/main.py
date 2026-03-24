from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
from app.api.department import router as departments_router
from app.api.ai import router as ai_router
from app.schemas.common import ErrorDetail, ErrorResponse

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
app.include_router(departments_router, prefix="/departments", tags=["departments"])
app.include_router(departments_router, prefix="/departements", tags=["departements"])
app.include_router(ai_router)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    payload = ErrorResponse(detail=ErrorDetail(**detail))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = ErrorResponse(
        detail=ErrorDetail(code="validation_error", message="Request validation failed"),
        errors=exc.errors(),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception path=%s", request.url.path)
    payload = ErrorResponse(
        detail=ErrorDetail(code="internal_error", message="Internal server error"),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())

@app.get("/")
def health_check():
    return {"status": "ok"}
