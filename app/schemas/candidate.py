from pydantic import BaseModel
from typing import List

class CandidateUploadRespose(BaseModel):
    filename: str
    skills: List[str]
    preview: str