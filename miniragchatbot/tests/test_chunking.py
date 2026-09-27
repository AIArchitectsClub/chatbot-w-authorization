from app.ingestion.chunk import chunk_pages, document_title
from app.ingestion.extract import PageText

LEAVE_PAGE = """Acme Corp - Leave Policy
1. Overview
This policy applies to all full-time employees.
2. Annual Leave Entitlement
Full-time employees are entitled to 22 working days of paid annual leave per calendar year.
Unused leave of up to 5 days may be carried forward.
"""


def chunks(text: str, max_chars=700, overlap=100, ocr=False):
    pages = [PageText(1, text, ocr=ocr)]
    return chunk_pages(pages, document_title(pages, "fallback"), max_chars, overlap)


def test_splits_at_numbered_headings_and_prefixes_title_and_section():
    result = chunks(LEAVE_PAGE)
    assert [c.section for c in result] == ["1. Overview", "2. Annual Leave Entitlement"]
    assert result[1].text.startswith("Acme Corp - Leave Policy - 2. Annual Leave Entitlement\n")
    assert "22 working days" in result[1].text


def test_hard_wrapped_lines_are_rejoined():
    assert "\n" not in chunks(LEAVE_PAGE)[1].text.split("\n", 1)[1]


def test_paragraphs_without_headings_are_chunked_on_blank_lines():
    text = "Strategic Plan\n\n" + "\n\n".join(f"Priority {n}: " + "word " * 60 for n in range(1, 5))
    result = chunks(text, max_chars=400)
    assert len(result) >= 2
    assert all(c.text.startswith("Strategic Plan\n") for c in result)
    assert all(len(c.text) < 500 for c in result)


def test_long_section_is_split_on_sentences_with_overlap():
    sentences = [f"Sentence number {n} ends here." for n in range(30)]
    text = "Doc Title\n1. Big Section\n" + " ".join(sentences)
    result = chunks(text, max_chars=200, overlap=60)

    assert len(result) > 1
    assert all(len(c.text.split("\n", 1)[1]) <= 200 for c in result)
    # overlap: the last sentence of a chunk reappears at the start of the next one
    first_tail = result[0].text.split("\n", 1)[1].split(". ")[-1]
    assert first_tail.rstrip(".") in result[1].text
