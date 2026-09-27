"""RBAC-filtered retrieval, plugged into the Agno agent as its `knowledge_retriever`."""

import logging
from typing import Any, Awaitable, Callable

from agno.vectordb.qdrant import Qdrant

from app.rag.rbac import build_filter
from app.schemas import UserClaims

logger = logging.getLogger(__name__)

Retriever = Callable[..., Awaitable[list[dict[str, Any]]]]


def make_rbac_retriever(
    user: UserClaims, vector_db: Qdrant, top_k: int, min_score: float
) -> Retriever:
    """Build a retriever bound to one user's permissions.

    Agno's built-in knowledge filters only support equality matches, so the
    range/any-of RBAC rule is passed to Qdrant as a native `Filter` instead.
    The filter is derived from the verified JWT claims, never from the model
    or the chat message.
    """
    query_filter = build_filter(user)

    async def retrieve(query: str, num_documents: int | None = None, **_: Any) -> list[dict[str, Any]]:
        embedding = await vector_db.embedder.async_get_embedding(query)
        response = await vector_db.async_client.query_points(
            collection_name=vector_db.collection,
            query=embedding,
            query_filter=query_filter,  # enforced inside Qdrant, at query time
            limit=num_documents or top_k,
            score_threshold=min_score,
            with_payload=True,
        )
        results = [
            {
                "source": point.payload["meta_data"]["source_file"],
                "page": point.payload["meta_data"]["page"],
                "content": point.payload["content"],
            }
            for point in response.points
            if point.payload
        ]
        logger.info(
            "retrieved %d chunk(s) for user=%s dept=%s level=%d: %s",
            len(results), user.sub, user.department, user.level,
            [f"{r['source']}:p{r['page']}" for r in results],
        )
        return results

    return retrieve
