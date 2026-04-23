import uuid

from app.api import candidates as candidates_api


UNKNOWN_TERM = "Quantum-Safe Cryptography"


def _fake_parsed_payload(*, email: str, unknown_term: str = UNKNOWN_TERM) -> dict:
    return {
        "ok": True,
        "degraded": False,
        "errors": [],
        "warnings": [],
        "text_length": 180,
        "skills": ["python", unknown_term],
        "skills_grouped": {
            "technical": ["python", unknown_term],
            "management": [],
            "business": [],
            "soft-skills": [],
            "other": [],
        },
        "skill_hierarchy": [],
        "skill_graph": {},
        "extracted_languages": ["english"],
        "language_details": [{"language": "english", "level": "B2", "source": "language_section:languages"}],
        "extraction_channels": {
            "catalog_match": ["python"],
            "open_vocab": [unknown_term],
            "soft_skill": [],
            "sentence": [],
            "semantic_augment": [],
            "language": ["english"],
            "certification": [],
            "hands_on_project": [],
            "project_text": [],
            "project_validated_skill": [],
        },
        "extracted_skills": [
            {
                "skill": "python",
                "confidence": 0.98,
                "confidence_normalized": 1.0,
                "source": "exact",
                "source_label": "catalog_match",
                "confidence_band": "high",
                "evidence": ["Python"],
            },
            {
                "skill": unknown_term,
                "confidence": 0.64,
                "confidence_normalized": 0.65,
                "source": "cv_section:skills",
                "source_label": "open_vocab",
                "confidence_band": "low",
                "evidence": [f"Skills: Python, {unknown_term}"],
            },
        ],
        "preview": f"Candidate mentions Python and {unknown_term}",
        "extracted_full_name": "Amina Review",
        "extracted_email": email,
        "extracted_phone": "+216 55 000 111",
        "predicted_title": "Security Engineer",
        "predicted_experience_years": 2.0,
        "certifications": [],
        "hands_on_projects": [],
        "project_skill_links": [],
    }


def _upload_cv(client, headers, monkeypatch, *, email: str, filename: str = "resume.pdf"):
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(email=email),
        raising=False,
    )
    return client.post(
        "/candidates/upload_cv",
        headers=headers,
        files={"file": (filename, b"%PDF-1.7\n%fake cv content", "application/pdf")},
    )


def test_upload_cv_queues_unknown_skill_for_review(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    email = f"amina.review.{marker}@example.com"

    upload = _upload_cv(client, admin_auth, monkeypatch, email=email, filename=f"amina_{marker}.pdf")
    assert upload.status_code == 200, upload.text
    body = upload.json()
    assert body["needs_review_count"] == 1
    assert body["queued_unknown_entities"] == [UNKNOWN_TERM]

    skills = client.get("/skills/", headers=admin_auth, params={"limit": 50})
    assert skills.status_code == 200, skills.text
    skill_names = {str(row["name"]).lower() for row in skills.json()}
    assert "python" in skill_names
    assert UNKNOWN_TERM.lower() not in skill_names

    candidates = client.get("/candidates/", headers=admin_auth, params={"search": marker, "limit": 20})
    assert candidates.status_code == 200, candidates.text
    rows = candidates.json()
    assert rows, "Expected uploaded candidate to be persisted"
    assert set(rows[0]["skills"]) == {"python"}

    unknowns = client.get("/learning/unknown-entities", headers=admin_auth, params={"status": "pending"})
    assert unknowns.status_code == 200, unknowns.text
    queue = unknowns.json()
    assert queue, "Expected one pending unknown entity"
    first = queue[0]
    assert first["raw_value"] == UNKNOWN_TERM
    assert first["status"] == "pending"
    assert first["entity_type_guess"] == "skill"
    assert first["occurrence_count"] >= 1


def test_admin_can_review_unknown_skill_and_future_upload_uses_new_skill(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    first_email = f"active.learning.{marker}@example.com"
    upload = _upload_cv(client, admin_auth, monkeypatch, email=first_email, filename=f"first_{marker}.pdf")
    assert upload.status_code == 200, upload.text

    unknowns = client.get("/learning/unknown-entities", headers=admin_auth, params={"search": UNKNOWN_TERM})
    assert unknowns.status_code == 200, unknowns.text
    queue = unknowns.json()
    assert queue, "Expected pending unknown entity before review"
    entity_id = int(queue[0]["id"])

    review = client.post(
        f"/learning/unknown-entities/{entity_id}/review",
        headers=admin_auth,
        json={
            "decision": "approved",
            "entity_type": "skill",
            "canonical_value": UNKNOWN_TERM,
            "notes": "Validated by recruiter review",
        },
    )
    assert review.status_code == 200, review.text
    reviewed = review.json()
    assert reviewed["status"] == "approved"
    assert reviewed["resolved_entity_type"] == "skill"
    assert reviewed["canonical_skill_id"] is not None
    assert reviewed["reviews"]
    assert reviewed["reviews"][0]["decision"] == "approved"

    skills = client.get("/skills/", headers=admin_auth, params={"search": "quantum-safe", "limit": 20})
    assert skills.status_code == 200, skills.text
    skill_names = {str(row["name"]).lower() for row in skills.json()}
    assert UNKNOWN_TERM.lower() in skill_names

    second_marker = uuid.uuid4().hex[:8]
    second_email = f"active.learning.second.{second_marker}@example.com"
    second_upload = _upload_cv(client, admin_auth, monkeypatch, email=second_email, filename=f"second_{second_marker}.pdf")
    assert second_upload.status_code == 200, second_upload.text
    second_body = second_upload.json()
    assert second_body["needs_review_count"] == 0
    assert second_body["queued_unknown_entities"] == []

    candidates = client.get("/candidates/", headers=admin_auth, params={"search": second_marker, "limit": 20})
    assert candidates.status_code == 200, candidates.text
    rows = candidates.json()
    assert rows, "Expected second uploaded candidate to be persisted"
    assert set(rows[0]["skills"]) == {"python", UNKNOWN_TERM}


def test_learning_endpoints_require_admin_policy(client, admin_auth, user_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    email = f"policy.learning.{marker}@example.com"
    upload = _upload_cv(client, admin_auth, monkeypatch, email=email, filename=f"policy_{marker}.pdf")
    assert upload.status_code == 200, upload.text

    forbidden_list = client.get("/learning/unknown-entities", headers=user_auth)
    assert forbidden_list.status_code == 403

    unknowns = client.get("/learning/unknown-entities", headers=admin_auth, params={"search": UNKNOWN_TERM})
    assert unknowns.status_code == 200, unknowns.text
    entity_id = int(unknowns.json()[0]["id"])

    forbidden_review = client.post(
        f"/learning/unknown-entities/{entity_id}/review",
        headers=user_auth,
        json={"decision": "rejected", "entity_type": "unknown"},
    )
    assert forbidden_review.status_code == 403
