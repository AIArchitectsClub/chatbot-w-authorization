"""PDF text extraction with automatic OCR fallback for scanned pages."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf

logger = logging.getLogger(__name__)

# A page with fewer extractable characters than this is treated as a scan.
MIN_TEXT_CHARS = 30
OCR_DPI = 200
# An OCR line gap above this multiple of the page's median gap starts a paragraph.
PARAGRAPH_GAP_FACTOR = 2.0


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str
    ocr: bool


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "warning"})


def _ocr_page(page: pymupdf.Page) -> str:
    """Rasterise a page and OCR it, keeping paragraph breaks as blank lines.

    OCR returns lines, not paragraphs. A line whose vertical gap to the previous
    line is well above the page's typical line gap starts a new paragraph, which
    adapts to whatever font size and spacing the scan uses.
    """
    pix = page.get_pixmap(dpi=OCR_DPI)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    result = _ocr_engine()(image)
    if result.txts is None or result.boxes is None or len(result.txts) == 0:
        return ""

    # boxes: 4 corner points per line, top-left first; sort lines top to bottom.
    lines = sorted(zip(result.boxes, result.txts), key=lambda item: item[0][0][1])
    gaps = [
        float(cur_box[0][1]) - float(prev_box[2][1])  # top of this line - bottom of previous
        for (prev_box, _), (cur_box, _) in zip(lines, lines[1:])
    ]
    threshold = PARAGRAPH_GAP_FACTOR * float(np.median(gaps)) if gaps else float("inf")

    out = [lines[0][1]]
    for gap, (_, text) in zip(gaps, lines[1:]):
        if gap > threshold:
            out.append("")  # blank line => paragraph break
        out.append(text)
    return "\n".join(out)


def extract_pages(path: Path) -> list[PageText]:
    """Return text per page; pages without a text layer are OCR'd automatically."""
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_CHARS:
                pages.append(PageText(index, text, ocr=False))
                continue
            logger.info("%s p%d has no text layer; running OCR", path.name, index)
            pages.append(PageText(index, _ocr_page(page).strip(), ocr=True))
    return pages
