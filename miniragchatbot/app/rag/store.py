"""Construction of the Agno vector store (Qdrant) and embedder."""

from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.vectordb.qdrant import Qdrant

from app.config import Settings


def make_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        id=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        normalize_embeddings=True,
    )


def make_vector_db(settings: Settings, embedder: SentenceTransformerEmbedder) -> Qdrant:
    """Qdrant server if QDRANT_URL is set, otherwise embedded on local disk."""
    if settings.qdrant_url:
        return Qdrant(
            collection=settings.qdrant_collection,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            embedder=embedder,
        )
    return Qdrant(
        collection=settings.qdrant_collection,
        path=str(settings.qdrant_path),
        embedder=embedder,
    )
