"""Role-based access control expressed as a Qdrant payload filter.

The filter is sent with the vector query, so the database only ever considers
chunks the user may see. Restricted text never leaves Qdrant.
"""

from qdrant_client import models

from app.schemas import UserClaims

# Anyone may read hr content (still bounded by their access level).
OPEN_DEPARTMENT = "hr"

# Agno stores chunk metadata under the `meta_data` key of each Qdrant payload.
DEPARTMENT_KEY = "meta_data.department"
ACCESS_LEVEL_KEY = "meta_data.access_level"


def allowed_departments(user: UserClaims) -> list[str]:
    return sorted({user.department, OPEN_DEPARTMENT})


def build_filter(user: UserClaims) -> models.Filter:
    """access_level <= user.level AND department IN {user.department, 'hr'}."""
    return models.Filter(
        must=[
            models.FieldCondition(key=ACCESS_LEVEL_KEY, range=models.Range(lte=user.level)),
            models.FieldCondition(
                key=DEPARTMENT_KEY, match=models.MatchAny(any=allowed_departments(user))
            ),
        ]
    )
