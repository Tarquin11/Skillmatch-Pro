from app.schemas.candidate import CandidateUploadRespose

def test_candidate_upload_response_contract_keys():
    obj = CandidateUploadRespose(filename="x.pdf")
    data = obj.model_dump()

    expected = {
        "filename", "ok", "degraded", "errors", "warnings", "text_length",
        "skills", "skills_grouped", "skill_hierarchy", "skill_graph", "extracted_languages", "language_details", "extraction_channels", "preview", "extracted_skills",
        "extracted_full_name", "extracted_email", "extracted_phone",
        "predicted_title", "predicted_experience_years",
        "certifications", "hands_on_projects", "project_skill_links",
    }
    assert set(data.keys()) == expected
    assert isinstance(data["errors"], list)
    assert isinstance(data["warnings"], list)
    assert isinstance(data["skills"], list)
    assert isinstance(data["skills_grouped"], dict)
    assert isinstance(data["skill_hierarchy"], list)
    assert isinstance(data["skill_graph"], dict)
    assert isinstance(data["extracted_languages"], list)
    assert isinstance(data["language_details"], list)
    assert isinstance(data["extraction_channels"], dict)
    assert isinstance(data["extracted_skills"], list)
