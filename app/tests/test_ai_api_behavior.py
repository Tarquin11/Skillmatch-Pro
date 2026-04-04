import os
import time
import uuid
from typing import Any

from app.api import match as match_api
from app.schemas.match import JobMatchResponse


def _seed_employees(client, admin_auth, count: int = 10) -> None:
    for idx in range(count):
        uid = uuid.uuid4().hex[:8]
        payload = {
            "employeeNumber": f"EMP-{uid}",
            "first_name": f"AI{idx}",
            "last_name": "Behavior",
            "full_name": f"AI{idx} Behavior",
            "email": f"ai_behavior_{uid}@example.com",
            "departement": "IT",
            "position": "Software Engineer" if idx % 2 == 0 else "Data Analyst",
        }
        r = client.post("/employees/", headers=admin_auth, json=payload)
        assert r.status_code == 201, r.text


def _match_payload(limit: int = 8) -> dict[str, Any]:
    return {
        "job_title": "Backend Engineer",
        "required_skills": ["python", "sql", "docker"],
        "min_experience": 0,
        "limit": limit,
    }


def _call_match(client, admin_auth, payload: dict[str, Any], path: str = "/match/job") -> tuple[Any, float]:
    started = time.perf_counter()
    resp = client.post(path, headers=admin_auth, json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return resp, elapsed_ms


def _ranking_signature(body: dict[str, Any]) -> list[tuple[int, float, str]]:
    return [
        (
            int(row["employee_id"]),
            float(row["predicted_fit_score"]),
            str(row["scoring_source"]),
        )
        for row in body.get("results", [])
    ]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank percentile.
    rank = max(1, int(round(0.95 * len(ordered))))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def test_match_job_schema_and_alias_consistency(client, admin_auth, monkeypatch):
    monkeypatch.setattr(match_api.inference_service, "_DRIFT_MONITORING_ENABLED", False, raising=False)
    _seed_employees(client, admin_auth, count=8)
    payload = _match_payload(limit=6)

    r_job, _ = _call_match(client, admin_auth, payload, path="/match/job")
    assert r_job.status_code == 200, r_job.text
    body_job = r_job.json()
    JobMatchResponse(**body_job)

    r_jobs, _ = _call_match(client, admin_auth, payload, path="/match/jobs")
    assert r_jobs.status_code == 200, r_jobs.text
    body_jobs = r_jobs.json()
    JobMatchResponse(**body_jobs)

    assert [row["employee_id"] for row in body_job["results"]] == [
        row["employee_id"] for row in body_jobs["results"]
    ]
    assert len(body_job["results"]) <= payload["limit"]


def test_match_job_ranking_is_stable_for_identical_requests(client, admin_auth, monkeypatch):
    monkeypatch.setattr(match_api.inference_service, "_DRIFT_MONITORING_ENABLED", False, raising=False)
    _seed_employees(client, admin_auth, count=12)
    payload = _match_payload(limit=10)

    signatures: list[list[tuple[int, float, str]]] = []
    for _ in range(3):
        r, _ = _call_match(client, admin_auth, payload)
        assert r.status_code == 200, r.text
        body = r.json()
        JobMatchResponse(**body)
        signatures.append(_ranking_signature(body))

        scores = [float(item["predicted_fit_score"]) for item in body["results"]]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    assert signatures[0] == signatures[1] == signatures[2]


def test_match_job_latency_is_reasonable_after_warmup(client, admin_auth, monkeypatch):
    monkeypatch.setattr(match_api.inference_service, "_DRIFT_MONITORING_ENABLED", False, raising=False)
    _seed_employees(client, admin_auth, count=16)
    payload = _match_payload(limit=10)

    max_p95_ms = float(os.getenv("TEST_MATCH_P95_MS", "1500"))
    max_single_ms = float(os.getenv("TEST_MATCH_MAX_MS", "2500"))

    # Warm-up request (model + caches).
    warmup, _ = _call_match(client, admin_auth, payload)
    assert warmup.status_code == 200, warmup.text

    samples_ms: list[float] = []
    for _ in range(5):
        r, elapsed_ms = _call_match(client, admin_auth, payload)
        assert r.status_code == 200, r.text
        samples_ms.append(elapsed_ms)

    assert _p95(samples_ms) <= max_p95_ms
    assert max(samples_ms) <= max_single_ms
