"""Ingest ./docs into the vector store.

Usage (from the project root):
    python -m scripts.ingest
"""

import logging

from app.config import get_settings
from app.ingestion.pipeline import ingest
from app.tls import configure_tls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    configure_tls(settings)
    result = ingest(settings)
    print(
        f"Done: {result.chunks_written} chunk(s) written "
        f"({result.files_changed} changed, {result.files_unchanged} unchanged, "
        f"{result.files_removed} removed)."
    )


if __name__ == "__main__":
    main()
