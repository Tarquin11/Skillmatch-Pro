from typing import Final

class Policy:
    EMPLOYEE_WRITE: Final[str] = "employee:write"
    JOB_WRITE: Final[str] = "job:write"
    JOB_SKILL_WRITE: Final[str] = "job_skill:write"
    SKILL_WRITE: Final[str] = "skill:write"
    EMPLOYEE_SKILL_WRITE: Final[str] = "employee_skill:write"
    DEPARTMENT_WRITE: Final[str] = "department:write"

POLICY_TO_ROLES: dict[str, frozenset[str]] = {
    Policy.EMPLOYEE_WRITE: frozenset({"admin"}),
    Policy.JOB_WRITE: frozenset({"admin"}),
    Policy.JOB_SKILL_WRITE: frozenset({"admin"}),
    Policy.SKILL_WRITE: frozenset({"admin"}),
    Policy.EMPLOYEE_SKILL_WRITE: frozenset({"admin"}),
    Policy.DEPARTMENT_WRITE: frozenset({"admin"}),
}

def roles_for_policy(policy: str) -> frozenset[str]:
    roles = POLICY_TO_ROLES.get(policy)
    if roles is None:
        raise KeyError(f"Unknown RBAC policy: {policy}")
    return roles
