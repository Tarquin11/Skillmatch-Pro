from fastapi import APIRouter, HTTPException , UploadFile, File, Depends, status
from sqlalchemy.orm import Session
import io
import zipfile
from app.schemas.candidate import CandidateUploadRespose
from app.schemas.common import ErrorResponse
from app.services.cv_parser import extract_text, detect_skills , detect_skills_with_confidence
from app.api.auth import get_current_active_user
from app.db.database import get_db
from app.models.skill import Skill

router = APIRouter(prefix="/candidates", tags=["Candidates"], dependencies=[Depends(get_current_active_user)])

_ALLOWED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

def _sniff_cv_mime(file_bytes: bytes) -> str:
    if not file_bytes:
        return "application/octet-stream"
    if file_bytes.startswith(b"%PDF-"):
        return "application/pdf"
    if file_bytes[:4] in (b"PK\x03\x04",b"PK\x05\x06",b"PK\x07\x08"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                names = {n.lower() for n in zf.namelist()}
                if "[content_types].xml" in names and "word/document.xml" in names:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        except Exception:
            pass
        return "application/octet-stream"

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
        sniffed = _sniff_cv_mime(contents)
        if sniffed not in _ALLOWED_MIMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_file_type","message":"Seulement Les Fichiers PDF et DOCX sont supportés !  / Only PDF and DOCX files are supported !"}
            )
        if sniffed == "application/pdf":
            safe_name = file.filename or "cv.pdf"
            if not safe_name.lower().endswith(".pdf"):
                safe_name = "cv.pdf"
        else:
            safe_name = file.filename or "cv.docx"
            if not safe_name.lower().endswith(".docx"):
                safe_name = "cv.docx"
        text = extract_text(contents, safe_name)
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

    
