import uuid

from app.api import candidates as candidates_api


UNKNOWN_TERM = "Quantum-Safe Cryptography"


def _fake_parsed_payload(
    *,
    email: str,
    unknown_term: str = UNKNOWN_TERM,
    include_unknown_skill: bool = True,
    certifications: list[str] | None = None,
    projects: list[str] | None = None,
) -> dict:
    skills = ["python"] + ([unknown_term] if include_unknown_skill else [])
    extracted_skills = [
        {
            "skill": "python",
            "confidence": 0.98,
            "confidence_normalized": 1.0,
            "source": "exact",
            "source_label": "catalog_match",
            "confidence_band": "high",
            "evidence": ["Python"],
        }
    ]
    if include_unknown_skill:
        extracted_skills.append(
            {
                "skill": unknown_term,
                "confidence": 0.64,
                "confidence_normalized": 0.65,
                "source": "cv_section:skills",
                "source_label": "open_vocab",
                "confidence_band": "low",
                "evidence": [f"Skills: Python, {unknown_term}"],
            }
        )

    return {
        "ok": True,
        "degraded": False,
        "errors": [],
        "warnings": [],
        "text_length": 180,
        "skills": skills,
        "skills_grouped": {
            "technical": skills,
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
            "open_vocab": [unknown_term] if include_unknown_skill else [],
            "soft_skill": [],
            "sentence": [],
            "semantic_augment": [],
            "language": ["english"],
            "certification": certifications or [],
            "hands_on_project": projects or [],
            "project_text": projects or [],
            "project_validated_skill": [],
        },
        "extracted_skills": extracted_skills,
        "preview": f"Candidate mentions Python and {unknown_term}",
        "extracted_full_name": "Amina Review",
        "extracted_email": email,
        "extracted_phone": "+216 55 000 111",
        "predicted_title": "Security Engineer",
        "predicted_experience_years": 2.0,
        "certifications": certifications or [],
        "hands_on_projects": projects or [],
        "project_skill_links": [],
    }


def _upload_cv(
    client,
    headers,
    monkeypatch,
    *,
    email: str,
    filename: str = "resume.pdf",
    unknown_term: str = UNKNOWN_TERM,
    include_unknown_skill: bool = True,
    certifications: list[str] | None = None,
    projects: list[str] | None = None,
):
    monkeypatch.setattr(
        candidates_api,
        "parse_cv_safe",
        lambda **_: _fake_parsed_payload(
            email=email,
            unknown_term=unknown_term,
            include_unknown_skill=include_unknown_skill,
            certifications=certifications,
            projects=projects,
        ),
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


def test_admin_can_review_unknown_skill_with_canonical_alias(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    raw_term = "Quantum Safe Crypto"
    canonical_term = "Quantum-Safe Cryptography"
    first_email = f"alias.learning.{marker}@example.com"
    upload = _upload_cv(
        client,
        admin_auth,
        monkeypatch,
        email=first_email,
        filename=f"alias_first_{marker}.pdf",
        unknown_term=raw_term,
    )
    assert upload.status_code == 200, upload.text

    unknowns = client.get("/learning/unknown-entities", headers=admin_auth, params={"search": raw_term})
    assert unknowns.status_code == 200, unknowns.text
    entity_id = int(unknowns.json()[0]["id"])

    review = client.post(
        f"/learning/unknown-entities/{entity_id}/review",
        headers=admin_auth,
        json={
            "decision": "approved",
            "entity_type": "skill",
            "canonical_value": canonical_term,
            "notes": "Use canonical taxonomy wording",
        },
    )
    assert review.status_code == 200, review.text

    second_marker = uuid.uuid4().hex[:8]
    second_email = f"alias.learning.second.{second_marker}@example.com"
    second_upload = _upload_cv(
        client,
        admin_auth,
        monkeypatch,
        email=second_email,
        filename=f"alias_second_{second_marker}.pdf",
        unknown_term=raw_term,
    )
    assert second_upload.status_code == 200, second_upload.text
    second_body = second_upload.json()
    assert second_body["needs_review_count"] == 0
    assert second_body["queued_unknown_entities"] == []

    candidates = client.get("/candidates/", headers=admin_auth, params={"search": second_marker, "limit": 20})
    assert candidates.status_code == 200, candidates.text
    rows = candidates.json()
    assert rows, "Expected second uploaded candidate to be persisted"
    assert set(rows[0]["skills"]) == {"python", canonical_term}


def test_reviewed_project_is_not_queued_again(client, admin_auth, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    project_name = "Zero Trust Migration Lab"
    first_email = f"project.learning.{marker}@example.com"
    upload = _upload_cv(
        client,
        admin_auth,
        monkeypatch,
        email=first_email,
        filename=f"project_first_{marker}.pdf",
        include_unknown_skill=False,
        projects=[project_name],
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["needs_review_count"] == 1

    unknowns = client.get("/learning/unknown-entities", headers=admin_auth, params={"search": project_name})
    assert unknowns.status_code == 200, unknowns.text
    queue = unknowns.json()
    assert queue, "Expected project to be queued before review"
    entity_id = int(queue[0]["id"])

    review = client.post(
        f"/learning/unknown-entities/{entity_id}/review",
        headers=admin_auth,
        json={
            "decision": "approved",
            "entity_type": "project",
            "canonical_value": project_name,
        },
    )
    assert review.status_code == 200, review.text
    assert review.json()["status"] == "approved"

    second_marker = uuid.uuid4().hex[:8]
    second_email = f"project.learning.second.{second_marker}@example.com"
    second_upload = _upload_cv(
        client,
        admin_auth,
        monkeypatch,
        email=second_email,
        filename=f"project_second_{second_marker}.pdf",
        include_unknown_skill=False,
        projects=[project_name],
    )
    assert second_upload.status_code == 200, second_upload.text
    assert second_upload.json()["needs_review_count"] == 0


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
