from app.schemas.candidate import CandidateUploadRespose

def test_candidate_upload_response_contract_keys():
    obj = CandidateUploadRespose(filename="x.pdf")
    data = obj.model_dump()

    expected = {
        "filename", "ok", "degraded", "errors", "warnings", "text_length",
        "skills", "preview", "extracted_skills",
        "predicted_title", "predicted_experience_years",
    }
    assert set(data.keys()) == expected
    assert isinstance(data["errors"], list)
    assert isinstance(data["warnings"], list)
    assert isinstance(data["skills"], list)
    assert isinstance(data["extracted_skills"], list)
