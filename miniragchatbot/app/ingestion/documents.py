"""Which documents exist, and who may read them."""

from dataclasses import dataclass
from pathlib import Path

from app.schemas import Department

# Explicit allow-list: (department, file) -> access_level.
# A PDF that is not listed here is refused rather than given a default level,
# so a new document can never become readable by accident (fail closed).
ACCESS_LEVELS: dict[tuple[Department, str], int] = {
    ("hr", "leave_policy.pdf"): 1,
    ("hr", "code_of_conduct.pdf"): 1,
    ("hr", "performance_review.pdf"): 2,
    ("finance", "expense_policy.pdf"): 1,
    ("finance", "travel_reimbursement.pdf"): 1,
    ("exec", "compensation_committee.pdf"): 3,
    ("exec", "strategic_plan.pdf"): 3,
}


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    department: Department
    access_level: int


def discover_documents(docs_dir: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in sorted(docs_dir.glob("*/*.pdf")):
        department = path.parent.name
        level = ACCESS_LEVELS.get((department, path.name))  # type: ignore[arg-type]
        if level is None:
            raise ValueError(
                f"{department}/{path.name} has no access_level in ACCESS_LEVELS; "
                "add it explicitly before ingesting."
            )
        documents.append(SourceDocument(path, department, level))  # type: ignore[arg-type]
    return documents
