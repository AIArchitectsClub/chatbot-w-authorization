"""Structure-aware chunking.

Strategy (see README_SUBMISSION.md for the reasoning):
  1. Split each page at numbered headings ("2. Annual Leave Entitlement").
     A policy section is a self-contained unit of meaning, so it is the
     natural retrieval unit.
  2. Inside a section, pack whole paragraphs, and only fall back to sentences
     when a paragraph alone exceeds the size limit.
  3. When one section spans several chunks, carry the trailing sentence(s)
     into the next chunk (character-bounded overlap) so an answer that
     straddles the boundary is still retrievable.
  4. Prefix every chunk with "<document title> - <section heading>" so a chunk
     is understandable, and embeds well, without its neighbours.
"""

import re
from dataclasses import dataclass

from app.ingestion.extract import PageText

HEADING_RE = re.compile(r"^\d+\.\s+\S.*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    page: int
    section: str | None
    text: str  # includes the "title - section" prefix


def document_title(pages: list[PageText], fallback: str) -> str:
    """The first non-empty line of the document."""
    for page in pages:
        for line in page.text.splitlines():
            if line.strip():
                return line.strip()
    return fallback


def _split_sections(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in lines:
        if HEADING_RE.match(line.strip()):
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)
    return [(heading, body) for heading, body in sections if any(l.strip() for l in body)]


def _paragraphs(body: list[str]) -> list[str]:
    """Blank lines separate paragraphs; hard line-wraps inside one are joined."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in body:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _units(paragraphs: list[str], max_chars: int) -> list[str]:
    """Paragraphs, with any over-long paragraph broken into sentences."""
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(SENTENCE_RE.split(paragraph))
    return units


def _overlap_tail(chunk_units: list[str], overlap_chars: int) -> list[str]:
    tail: list[str] = []
    size = 0
    for unit in reversed(chunk_units):
        if size + len(unit) > overlap_chars:
            break
        tail.insert(0, unit)
        size += len(unit)
    return tail


def _pack(units: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = " ".join([*current, unit])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [*_overlap_tail(current, overlap_chars), unit]
        else:
            current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: list[PageText], title: str, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        lines = [l for l in page.text.splitlines() if l.strip() != title]
        for heading, body in _split_sections(lines):
            units = _units(_paragraphs(body), max_chars)
            prefix = f"{title} - {heading}" if heading else title
            for text in _pack(units, max_chars, overlap_chars):
                chunks.append(Chunk(page.page, heading, f"{prefix}\n{text}"))
    return chunks
