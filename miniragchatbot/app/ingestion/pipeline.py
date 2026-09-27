"""Ingestion: PDFs -> text (OCR if needed) -> chunks -> embeddings -> Qdrant."""

import hashlib
import logging
from dataclasses import dataclass

from agno.knowledge.document import Document
from agno.vectordb.qdrant import Qdrant
from qdrant_client import models

from app.config import Settings
from app.ingestion.chunk import chunk_pages, document_title
from app.ingestion.documents import SourceDocument, discover_documents
from app.ingestion.extract import extract_pages
from app.rag.rbac import ACCESS_LEVEL_KEY, DEPARTMENT_KEY
from app.rag.store import make_embedder, make_vector_db

logger = logging.getLogger(__name__)


def build_documents(source: SourceDocument, settings: Settings) -> list[Document]:
    """One Agno Document per chunk, carrying the RBAC metadata."""
    pages = extract_pages(source.path)
    title = document_title(pages, fallback=source.path.stem)
    chunks = chunk_pages(pages, title, settings.chunk_max_chars, settings.chunk_overlap_chars)
    ocr_pages = {p.page for p in pages if p.ocr}

    return [
        Document(
            id=f"{source.department}/{source.path.name}#{index}",
            name=source.path.name,
            content=chunk.text,
            meta_data={
                "department": source.department,
                "access_level": source.access_level,
                "source_file": source.path.name,
                "page": chunk.page,
                "section": chunk.section,
                "ocr": chunk.page in ocr_pages,
            },
        )
        for index, chunk in enumerate(chunks)
    ]


def _create_payload_indexes(vector_db: Qdrant) -> None:
    """Index the RBAC fields so filtered search stays fast on a Qdrant server."""
    vector_db.client.create_payload_index(
        vector_db.collection, DEPARTMENT_KEY, models.PayloadSchemaType.KEYWORD
    )
    vector_db.client.create_payload_index(
        vector_db.collection, ACCESS_LEVEL_KEY, models.PayloadSchemaType.INTEGER
    )


def _existing_file_hashes(vector_db: Qdrant) -> dict[str, set[str]]:
    """Map ``source_file`` -> the content hash(es) currently stored for it.

    One scroll over the whole collection (cheap at this corpus size) answers both
    "which files are already here" and "is this exact file, by name AND content,
    already ingested" -- scoped by filename. Agno's own ``content_hash_exists``
    checks the hash alone, so it would wrongly call a renamed file "unchanged"
    just because its bytes match something already stored under a different name.
    """
    if not vector_db.exists():
        return {}

    hashes_by_file: dict[str, set[str]] = {}
    offset = None
    while True:
        points, offset = vector_db.client.scroll(
            collection_name=vector_db.collection,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for point in points:
            name = point.payload["meta_data"]["source_file"]
            hashes_by_file.setdefault(name, set()).add(point.payload["content_hash"])
        if offset is None:
            break
    return hashes_by_file


@dataclass(frozen=True)
class IngestResult:
    chunks_written: int
    files_changed: int
    files_unchanged: int
    files_removed: int


def ingest(settings: Settings) -> IngestResult:
    """Sync the collection with ./docs: only new or changed files are (re-)embedded.

    A file counts as unchanged only when a point already exists for that exact
    ``source_file`` AND content hash together (see ``_existing_file_hashes``) --
    checking the hash alone, as Agno's own ``content_hash_exists`` does, would
    wrongly call a renamed file "unchanged" just because its bytes match content
    already stored under a different name. A changed file's old chunks are
    cleared by ``source_file`` before inserting the new ones, since the old
    chunks live under a different, now-orphaned hash that a plain hash-keyed
    replace would never find. A file removed from disk has its chunks cleared
    the same way, so a document that changed access level or disappeared can
    never linger in the index with stale permissions.
    """
    sources = discover_documents(settings.docs_dir)
    if not sources:
        raise SystemExit(f"No PDFs found under {settings.docs_dir}")

    vector_db = make_vector_db(settings, make_embedder(settings))
    existing = _existing_file_hashes(vector_db)
    if not vector_db.exists():
        vector_db.create()
        _create_payload_indexes(vector_db)

    discovered = {source.path.name: source for source in sources}
    removed = existing.keys() - discovered.keys()
    for name in removed:
        vector_db.delete_by_metadata({"source_file": name})
        logger.info("removed %s: no longer present under %s", name, settings.docs_dir)

    chunks_written = 0
    changed = 0
    unchanged = 0
    for source in sources:
        content_hash = hashlib.md5(source.path.read_bytes()).hexdigest()
        if content_hash in existing.get(source.path.name, set()):
            unchanged += 1
            continue

        # New file, or an existing one whose content changed. Clearing by file
        # name before inserting is a no-op for a genuinely new file, and removes
        # the stale chunks for a modified one that the hash check alone would miss.
        vector_db.delete_by_metadata({"source_file": source.path.name})
        documents = build_documents(source, settings)
        vector_db.upsert(content_hash=content_hash, documents=documents)
        chunks_written += len(documents)
        changed += 1
        logger.info(
            "ingested %s/%s: %d chunks (level %d)",
            source.department, source.path.name, len(documents), source.access_level,
        )
    vector_db.close()

    logger.info(
        "ingestion summary: %d file(s) changed (%d chunks), %d unchanged, %d removed",
        changed, chunks_written, unchanged, len(removed),
    )
    return IngestResult(chunks_written, changed, unchanged, len(removed))
