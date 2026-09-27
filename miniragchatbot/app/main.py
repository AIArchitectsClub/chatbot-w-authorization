"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.ws_chat import router as chat_router
from app.config import Settings, get_settings
from app.rag.store import make_embedder, make_vector_db
from app.tls import configure_tls

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_tls(settings)
        # Loading the embedding model is slow and blocking, so it happens once at
        # startup (not on the first user request) and is shared by all connections.
        app.state.settings = settings
        app.state.vector_db = make_vector_db(settings, make_embedder(settings))
        logger.info("ready: llm=%s collection=%s", settings.llm_provider, settings.qdrant_collection)
        yield
        await app.state.vector_db.async_client.close()

    app = FastAPI(title="Mini RAG Chatbot", lifespan=lifespan)
    app.include_router(chat_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "qdrant_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = create_app()
