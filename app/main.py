from fastapi import FastAPI
from app.api import auth
from app.api.employees import router as employees_router
from app.api.candidates import router as candidates_router
from app.api.match import router as match_router
from app.api.jobs import router as jobs_router
from app.api.skills import router as skills_router
from app.api.department import router as departements_router

app = FastAPI(title="SkillMatch Pro")
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
