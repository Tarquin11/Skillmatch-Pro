import uuid

def test_employees_crud(client, admin_auth):
    uid = uuid.uuid4().hex[:6]
    payload = {
        "employeeNumber": f"EMP-{uid}",
        "first_name": "Jude",
        "last_name": "Bellingham",
        "full_name": "Jude Bellingham",
        "email": f"emp_{uid}@example.com",
        "departement": "IT",
        "position": "IT Support Specialist",
    }

    r = client.post("/employees/", headers=admin_auth, json=payload)
    assert r.status_code == 201, r.text
    emp_id = r.json()["id"]

    r = client.get(f"/employees/{emp_id}", headers=admin_auth)
    assert r.status_code == 200

    r = client.put(f"/employees/{emp_id}", headers=admin_auth, json={"position": "System Administrator"})
    assert r.status_code == 200
    assert r.json()["position"] == "System Administrator"

    r = client.delete(f"/employees/{emp_id}", headers=admin_auth)
    assert r.status_code == 204

    r = client.get(f"/employees/{emp_id}", headers=admin_auth)
    assert r.status_code == 404


def test_jobs_crud(client, admin_auth):
    r = client.post("/jobs/", headers=admin_auth, json={"title": "DevOps Engineer", "departement": "IT"})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]

    r = client.get(f"/jobs/{job_id}", headers=admin_auth)
    assert r.status_code == 200

    r = client.put(f"/jobs/{job_id}", headers=admin_auth, json={"description": "Updated"})
    assert r.status_code == 200

    r = client.delete(f"/jobs/{job_id}", headers=admin_auth)
    assert r.status_code == 204


def test_skills_crud(client, admin_auth):
    r = client.post("/skills/", headers=admin_auth, json={"name": "Kubernetes"})
    assert r.status_code == 201, r.text
    skill_id = r.json()["id"]

    r = client.get(f"/skills/{skill_id}", headers=admin_auth)
    assert r.status_code == 200

    r = client.put(f"/skills/{skill_id}", headers=admin_auth, json={"name": "K8s"})
    assert r.status_code == 200
    assert r.json()["name"] == "K8s"

    r = client.delete(f"/skills/{skill_id}", headers=admin_auth)
    assert r.status_code == 204


def test_departements_crud(client, admin_auth):
    r = client.post("/departements/", headers=admin_auth, json={"name": "QA"})
    assert r.status_code == 201, r.text
    dep_id = r.json()["id"]

    r = client.get(f"/departements/{dep_id}", headers=admin_auth)
    assert r.status_code == 200

    r = client.put(f"/departements/{dep_id}", headers=admin_auth, json={"name": "Quality Assurance"})
    assert r.status_code == 200
    assert r.json()["name"] == "Quality Assurance"

    r = client.delete(f"/departements/{dep_id}", headers=admin_auth)
    assert r.status_code == 204
