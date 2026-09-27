"""Optional: trust the operating system's certificate store.

Needed on networks that intercept TLS with a corporate CA (the download of the
embedding model from Hugging Face fails there otherwise). Off by default.
"""

from app.config import Settings


def configure_tls(settings: Settings) -> None:
    if not settings.use_system_trust_store:
        return
    import truststore

    truststore.inject_into_ssl()
