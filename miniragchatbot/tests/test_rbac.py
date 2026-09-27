"""RBAC is enforced by Qdrant itself: run the real filter against a real (in-memory) Qdrant."""

import pytest
from qdrant_client import QdrantClient, models

from app.rag.rbac import build_filter
from app.schemas import UserClaims

# (department, access_level, source_file) - mirrors the assignment's table.
CORPUS = [
    ("hr", 1, "leave_policy.pdf"),
    ("hr", 1, "code_of_conduct.pdf"),
    ("hr", 2, "performance_review.pdf"),
    ("finance", 1, "expense_policy.pdf"),
    ("finance", 1, "travel_reimbursement.pdf"),
    ("exec", 3, "compensation_committee.pdf"),
    ("exec", 3, "strategic_plan.pdf"),
]

USERS = {
    "emp-001": UserClaims(sub="emp-001", department="hr", level=1),
    "mgr-002": UserClaims(sub="mgr-002", department="hr", level=2),
    "exec-003": UserClaims(sub="exec-003", department="exec", level=3),
}


@pytest.fixture(scope="module")
def client() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=models.VectorParams(size=2, distance="Cosine"))
    client.upsert(
        "t",
        [
            models.PointStruct(
                id=i,
                vector=[1.0, 0.1 * i],  # all similar: the filter, not similarity, decides
                payload={"meta_data": {"department": d, "access_level": lvl, "source_file": f}},
            )
            for i, (d, lvl, f) in enumerate(CORPUS)
        ],
    )
    return client


def visible_files(client: QdrantClient, user: UserClaims) -> set[str]:
    response = client.query_points(
        "t", query=[1.0, 0.0], query_filter=build_filter(user), limit=100, with_payload=True
    )
    return {p.payload["meta_data"]["source_file"] for p in response.points}


def test_level1_hr_employee_sees_only_level1_hr(client):
    assert visible_files(client, USERS["emp-001"]) == {"leave_policy.pdf", "code_of_conduct.pdf"}


def test_hr_manager_also_sees_level2_hr(client):
    assert visible_files(client, USERS["mgr-002"]) == {
        "leave_policy.pdf",
        "code_of_conduct.pdf",
        "performance_review.pdf",
    }


def test_exec_sees_exec_plus_hr_exception_but_not_finance(client):
    assert visible_files(client, USERS["exec-003"]) == {
        "leave_policy.pdf",
        "code_of_conduct.pdf",
        "performance_review.pdf",
        "compensation_committee.pdf",
        "strategic_plan.pdf",
    }


def test_finance_user_sees_finance_and_hr_at_their_level(client):
    finance_user = UserClaims(sub="fin-1", department="finance", level=1)
    assert visible_files(client, finance_user) == {
        "expense_policy.pdf",
        "travel_reimbursement.pdf",
        "leave_policy.pdf",
        "code_of_conduct.pdf",
    }


def test_hr_exception_never_bypasses_access_level(client):
    assert "performance_review.pdf" not in visible_files(client, USERS["emp-001"])
