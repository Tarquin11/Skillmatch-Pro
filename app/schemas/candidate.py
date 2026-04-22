from pydantic import Field
from typing import Dict, List
from app.schemas.common import StrictBaseModel

class CandidateSkillExtraction(StrictBaseModel):
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_normalized: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str
    source_label: str | None = None
    confidence_band: str | None = None
    evidence: List[str] = Field(default_factory=list)

class CandidateLanguageExtraction(StrictBaseModel):
    language: str
    level: str | None = None
    source: str | None = None

class CandidateExtractionChannels(StrictBaseModel):
    catalog_match: List[str] = Field(default_factory=list)
    open_vocab: List[str] = Field(default_factory=list)
    soft_skill: List[str] = Field(default_factory=list)
    sentence: List[str] = Field(default_factory=list)
    semantic_augment: List[str] = Field(default_factory=list)
    language: List[str] = Field(default_factory=list)
    certification: List[str] = Field(default_factory=list)
    hands_on_project: List[str] = Field(default_factory=list)
    project_text: List[str] = Field(default_factory=list)


class CandidateSkillHierarchyNode(StrictBaseModel):
    parent: str
    subskills: List[str] = Field(default_factory=list)


class CandidateSkillGraphNode(StrictBaseModel):
    children: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class CandidateUploadRespose(StrictBaseModel):
    filename: str
    ok: bool = True
    degraded: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    text_length: int = Field(default=0, ge=0)
    skills: List[str] = Field(default_factory=list)
    skills_grouped: Dict[str, List[str]] = Field(default_factory=dict)
    skill_hierarchy: List[CandidateSkillHierarchyNode] = Field(default_factory=list)
    skill_graph: Dict[str, CandidateSkillGraphNode] = Field(default_factory=dict)
    extracted_languages: List[str] = Field(default_factory=list)
    language_details: List[CandidateLanguageExtraction] = Field(default_factory=list)
    extraction_channels: CandidateExtractionChannels = Field(default_factory=CandidateExtractionChannels)
    preview: str = ""
    extracted_skills: List[CandidateSkillExtraction] = Field(default_factory=list)
    extracted_full_name: str | None = None
    extracted_email: str | None = None
    extracted_phone: str | None = None
    predicted_title: str | None = None
    predicted_experience_years: float | None = None
    certifications: List[str] = Field(default_factory=list)
    hands_on_projects: List[str] = Field(default_factory=list)


class CandidateListItem(StrictBaseModel):
    id: int
    employee_number: str
    full_name: str
    email: str
    phone: str | None = None
    predicted_title: str | None = None
    predicted_experience_years: float | None = None
    skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    hands_on_projects: List[str] = Field(default_factory=list)
    uploaded_at: str | None = None


class CandidateUpdateRequest(StrictBaseModel):
    full_name: str | None = None
    employee_number: str | None = None
    skills: List[str] | None = None
    certifications: List[str] | None = None
    hands_on_projects: List[str] | None = None
