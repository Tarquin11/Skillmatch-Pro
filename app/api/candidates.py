from fastapi import APIRouter, HTTPException , UploadFile, File, Depends, status
from sqlalchemy.orm import Session
from app.schemas.candidate import CandidateUploadRespose
from app.schemas.common import ErrorResponse
from app.services.cv_parser import extract_text, detect_skills , detect_skills_with_confidence
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.skill import Skill

router = APIRouter(prefix="/candidates", tags=["Candidates"], dependencies=[Depends(get_current_active_user)])

@router.post(
    "/upload_cv",
    response_model=CandidateUploadRespose,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_cv (file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail={"code": "invalid_file_type", "message": "Seulement Les Fichiers PDF et DOCX sont supportés !  / Only PDF and DOCX files are supported ! "},)
    try:
        contents = await file.read()
        text = extract_text(contents, file.filename)
        #no hardcoded skills
        known_skills = [name for(name,) in db.query(Skill.name).all() if name]
        extracted_skills = detect_skills_with_confidence(text, known_skills=known_skills)
        skills = [row["skill"] for row in extracted_skills]
        #fallback compatible
        if not skills:
            skills = detect_skills(text)
            extracted_skills=[
                {"skill": skill, "confidence" : 0.6, "source": "legacy"}
                for skill in skills
            ]
        return CandidateUploadRespose(
            filename=file.filename,
            skills=skills,
            extracted_skills=extracted_skills,
            preview=(text or "")[:200],
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "cv_processing_failed","message":"Failed to process CV file. / Échec du traitement du fichier CV."},
        )

    
