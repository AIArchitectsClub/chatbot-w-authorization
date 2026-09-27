# High-level plan: Mini RAG Chatbot Backend

Source: README.md (take-home assignment). Nothing built yet at time of writing.

## What is being graded

| Rubric area | How the plan meets it |
|---|---|
| WebSocket and streaming | Exact protocol from the README, with real token-by-token streaming from the LLM |
| Parallel tool call | `asyncio.gather` over 3 one-second mocks, with a test asserting elapsed time is under 1.5s |
| No blocking in `async def` | Embedding, OCR and PDF parsing go through `asyncio.to_thread`. The Qdrant, LLM and JWT paths use async clients |
| RBAC at the DB layer | Qdrant payload filter built from the user's claims and applied in the query itself |
| Chunking | A justified strategy, documented with numbers |
| Scanned PDF | Automatic detection and OCR, with no manual step |
| Error handling | Auth failures close the socket. LLM and tool errors send `{"type":"error"}` and keep the connection open |
| Code quality | Type hints, Pydantic models, small functions, no dead code |
| README thinking | `README_SUBMISSION.md` with all 8 required sections |

## Stack decisions

- **API:** FastAPI on uvicorn, with `pydantic-settings` for config.
- **Framework policy:** use Agno wherever it can do the job (agent, models, knowledge/vector DB, embedder, chunking, PDF reading, memory, streaming). Keep custom only what the framework cannot do or what the rubric checks directly: JWT auth, the WebSocket protocol, the `build_filter` RBAC function, the parallel tool body. LangChain is not used (avoids two overlapping frameworks).
- **Agent:** Agno `Agent` with one tool `get_employee_context`, run with `arun(stream=True)` and mapped to the `stream` / `done` protocol. Per-connection session id and Agno's built-in session history for conversation memory.
- **LLM:** Groq Llama 3.3 70B via Agno's Groq model class; Agno's Ollama class (Qwen) as the second option. Provider chosen by config; Agno's model interface is the pluggable seam.
- **Vector DB and embeddings:** Agno `Knowledge` with `Qdrant` vector DB and the `SentenceTransformerEmbedder` (`BAAI/bge-small-en-v1.5`).
- **RBAC retrieval:** the filter must be a native Qdrant `Filter` (range on `access_level`, any-of on `department`). Agno's dict-style knowledge filters may only support equality matching. TO VERIFY against current Agno docs. Fallback: a custom `knowledge_retriever` (or a search tool) that queries Qdrant with our own `Filter`, still applied in the DB query.
- **PDF and OCR:** Agno PDF reader for text-based pages, with per-chunk metadata. Pages with little or no text are rasterised and run through RapidOCR (pip-only, no Tesseract binary needed on Windows). TO VERIFY: whether Agno's reader exposes page numbers; otherwise use PyMuPDF for extraction and feed Agno `Document`s.
- **Chunking:** Agno chunking strategy (recursive/document), sizes tuned to headings/paragraphs. TO VERIFY available strategies.
- **Auth:** PyJWT, HS256, verifying `exp` and required claims.

## Proposed structure

```
app/
  main.py            # app factory, lifespan (init Qdrant/embedder once)
  config.py          # Settings
  auth.py            # JWT decode -> UserClaims
  api/ws_chat.py     # /ws/chat handler, protocol state machine
  schemas.py         # Pydantic models for all WS messages + claims
  rag/
    retriever.py     # embed query, RBAC-filtered Qdrant search
    rbac.py          # build_filter(user) -> qdrant Filter
    prompt.py        # system prompt, context formatting, citations
  llm/
    factory.py       # config -> Agno model (Groq / Ollama / ...)
    agent.py         # Agno Agent, tool registered, streaming helper
  tools/employee_context.py   # parallel gather
  ingestion/
    extract.py chunk.py embed.py pipeline.py
scripts/ingest.py    # CLI entry
tests/
README_SUBMISSION.md
```

## Phases

1. **Scaffold and config:** layout, dependencies, `.env` loading, Qdrant via docker-compose, small Dockerfile.
2. **Ingestion:** walk `docs/`, derive `department` from folder, `access_level` from an explicit mapping; extract (OCR fallback), chunk, embed, upsert. Idempotent via deterministic point IDs.
3. **RBAC retrieval:** payload indexes on `department` and `access_level`; build the filter.
4. **LLM layer:** provider interface, streaming, tool-call loop (execute tool, then stream final answer).
5. **Parallel tool:** `get_employee_context` using `asyncio.gather`.
6. **WebSocket handler:** auth handshake with timeout, Pydantic validation per message, `auth_failed` + close on bad token, per-turn error handling, short per-connection conversation memory (stretch).
7. **Tests:** RBAC filter matrix for the 3 users, tool timing, auth failure cases, WS protocol test with a fake LLM, ingestion metadata check.
8. **Docs and polish:** `README_SUBMISSION.md`; small, meaningful commits throughout.

## Open points / assumptions

1. **RBAC semantics.** Express as `access_level <= level AND department IN {user.department, "hr"}`.
   - `emp-001`: hr level 1 only.
   - `mgr-002`: hr levels 1 and 2.
   - `exec-003`: exec plus hr, not finance.
   - The HR exception still respects `access_level`. Document as an assumption.
2. **Scanned PDF.** README calls `strategic_plan.pdf` the scanned one; verify against the sample docs before finalising the OCR approach.
3. **Chunking.** Starting point: structure-aware splitting (headings/paragraphs), ~500 tokens, ~15% overlap, page numbers in metadata. Finalise after inspecting the real PDFs.
4. **Stretch.** Include conversation memory. Reranker and hybrid search only if time allows.

## Implementation notes (what changed vs. this plan once built)

- Built and verified end to end; see README_SUBMISSION.md for run instructions and design notes.
- Agno 3.0.11 verified against its source: its Qdrant filters are equality-only, so RBAC goes through
  a custom async `knowledge_retriever` that passes a native Qdrant `Filter` (range + any-of).
- OCR: `rapidocr` 3.x (bundled models). Older `rapidocr-onnxruntime` dropped spaces; EasyOCR misread digits.
- Qdrant runs embedded (`./qdrant_data`) by default because Docker was not available; server mode via `QDRANT_URL`.
- Added a retrieval similarity floor (`RETRIEVAL_MIN_SCORE=0.65`) so irrelevant chunks never reach the model.
- Default LLM is local Ollama `qwen2.5`; Groq/OpenAI selectable with `LLM_PROVIDER`.
- Verified: 27 automated tests pass; tool call measured at ~1.0s; RBAC confirmed for all three test users.
- Also verified against a real Qdrant server (`docker-compose.yml`, `QDRANT_URL=http://localhost:6333`),
  not just embedded mode: same 39 chunks, same RBAC/tool/OCR behaviour end to end.
- **Ingestion is incremental, not a full rebuild** (`app/ingestion/pipeline.py::ingest`). Rationale: a
  full drop+recreate re-embeds every file on every run, which is fine at 7 PDFs but doesn't scale once
  new PDFs land regularly (e.g. daily). Current behaviour, per file: unchanged (same name AND same
  content hash already stored) -> skipped entirely, no re-embed; new or modified -> old chunks cleared
  by `source_file` first, then re-embedded and inserted; present in Qdrant but no longer discovered on
  disk -> its chunks are removed. `IngestResult` reports counts for changed/unchanged/removed so a run
  is auditable.
  - **Correctness pitfall found and fixed while implementing this:** Agno's own `Qdrant.upsert()` does
    an internal dedup keyed on `content_hash` **alone** (not scoped by filename), and its
    `content_hash_exists()` helper has the same scope. Naively reusing that helper to decide "is this
    file unchanged" is wrong: two different files with identical byte content (or a renamed file with
    unchanged bytes) would be misjudged as already-ingested, and would never actually get stored under
    their own `source_file` identity. Worse, calling `upsert()` on genuinely duplicate content silently
    evicts the *other* file's points, since Agno treats identical content hash as the same document
    regardless of name. Fix: `_existing_file_hashes()` scrolls the collection once and builds a
    `source_file -> {content_hash}` map ourselves, so "unchanged" is scoped to (name, hash) together,
    matching real ingestion semantics rather than Agno's own document-dedup semantics. Confirmed via a
    live scroll of the Qdrant payload that `content_hash` is a top-level field (not nested under
    `meta_data`), unlike the RBAC fields.
  - Verified live against the real Qdrant server: adding a new PDF only embeds that file; modifying an
    existing PDF only replaces that file's chunks; deleting a PDF from `docs/` only removes that file's
    chunks -- all confirmed by point counts and `source_file` distribution before/after, with the other
    files' chunks byte-identical (same IDs) across the run.
