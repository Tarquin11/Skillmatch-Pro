from fastapi import APIRouter, HTTPException , UploadFile, File, Depends, status
from app.schemas.candidate import CandidateUploadRespose
from app.schemas.common import ErrorResponse
from app.services.cv_parser import extract_text, detect_skills
from app.api.auth import get_current_active_user

router = APIRouter(prefix="/candidates", tags=["Candidates"], dependencies=[Depends(get_current_active_user)])

@router.post(
    "/upload_cv",
    response_model=CandidateUploadRespose,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_cv (file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail={"code": "invalid_file_type", "message": "Seulement Les Fichiers PDF sont supportés !  / Only PDF files are supported ! "},)
    try:
        contents = await file.read()
        text = extract_text(contents, file.filename)
        skills = detect_skills(text)

        return CandidateUploadRespose(
            filename=file.filename,
            skills=skills,
            preview=(text or "")[:200],
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "cv_processing_failed","message":"Failed to process CV file. / Échec du traitement du fichier CV."},
        )

    