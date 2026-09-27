# Mini RAG Chatbot Backend — Submission

A FastAPI backend that serves a policy chatbot over WebSocket (`/ws/chat`). It answers
from the PDFs in `./docs` using RAG, enforces role-based access **inside the vector
database query**, and can call a `get_employee_context` tool whose three lookups run in
parallel.

**Stack:** FastAPI · [Agno](https://docs.agno.com) (agent, model adapters, vector-DB and
embedder wrappers) · Qdrant · `BAAI/bge-small-en-v1.5` (sentence-transformers) ·
PyMuPDF + RapidOCR · PyJWT.

---

## 1. Run the ingestion script

Tested on Python 3.12. (I did not try 3.14: many ML wheels lag on the newest Python.)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env              # then set JWT_SECRET; the defaults use local Ollama

python -m scripts.ingest
```

Expected output: `Done: 39 chunks stored.` (7 PDFs; the scanned `strategic_plan.pdf` is
OCR'd automatically, with a log line saying so). The first run downloads the embedding
model (~130 MB).

Ingestion is a **full rebuild** (drop collection, re-create, re-insert), so it is
idempotent and a document whose access level changed can never linger with stale
permissions.

By default Qdrant runs **embedded** on local disk (`./qdrant_data`); no Docker needed.
That mode is single-process, so run ingestion **before** starting the server. To use a
real Qdrant server: `docker compose up -d`, set `QDRANT_URL=http://localhost:6333`, and
re-run ingestion.

## 2. Run the server

```bash
ollama pull qwen2.5               # default LLM, fully local, no API key
uvicorn app.main:app --port 8000
```

`GET /health` returns `{"status":"ok"}` once the embedding model has loaded.

To use Groq instead, set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env`.

## 3. Test it

**Browser client (the one provided):**

1. `python mint_tokens.py`, copy one of the three tokens.
2. Open `test_client.html`, paste the token, **Connect**, and chat.

**Command line (no browser):**

```bash
python -m scripts.ws_demo emp-001  "How many days of annual leave do I get?"
python -m scripts.ws_demo mgr-002  "How do performance ratings affect merit increases?"
python -m scripts.ws_demo exec-003 "What are the APAC priorities?" "Who is my manager?"
python -m scripts.ws_demo not-a-jwt "hello"      # -> auth_failed, socket closed
```

**Automated tests:** `pytest` (27 tests, ~25 s, no LLM or model download needed).

**Things worth trying, and what to expect:**

| As | Ask | Expected |
|----|-----|----------|
| emp-001 (hr, L1) | "How are performance ratings linked to bonus?" | Not found. `performance_review.pdf` is level 2 |
| mgr-002 (hr, L2) | same question | Answered from `performance_review.pdf` |
| exec-003 (exec, L3) | "What are the strategic priorities and APAC investment?" | Answered from the **scanned** PDF (USD 45 million) |
| any | "Who is my manager and how big is my team?" | Tool call, ~1 s: John Smith, team of 8 |
| any | "How much leave do I have given my grade?" | Tool call, then combined with the leave policy |

The server log prints one line per retrieval with the chunks that survived the RBAC
filter (`retrieved 0 chunk(s) for user=emp-001 ...`), which is the quickest way to *see*
the access control working.

Example exchange:

```
-> {"type":"auth","token":"<jwt>"}
<- {"type":"auth_success","user_id":"mgr-002","department":"hr","level":2}
-> {"type":"message","text":"Who is my manager?"}
<- {"type":"stream","text":"Your manager"}
<- {"type":"stream","text":" is John Smith."}
<- {"type":"done"}
```

---

## 4. Chunking strategy

**Structure-aware, not fixed-size.** Every source document is a short policy with
numbered sections ("2. Annual Leave Entitlement"), and each section is one self-contained
rule set. So:

1. Split each page at numbered headings. The section is the retrieval unit.
2. Inside a section, pack whole paragraphs up to **700 characters** (about 150–175
   tokens); only a paragraph that alone exceeds the limit is split, on sentence
   boundaries.
3. When one section does span several chunks, carry trailing sentences into the next
   chunk as **overlap, capped at 100 characters** (about 15%), so an answer straddling
   the cut is still retrievable. Overlap is *not* applied across section boundaries,
   since that would only blur unrelated topics.
4. Prefix every chunk with `<document title> - <section heading>`. A chunk reading
   "Employees are entitled to 22 days…" is ambiguous on its own; with the prefix it
   embeds and cites correctly.

**Why these numbers:** 700 characters is well inside bge-small's 512-token window, and
small enough that a chunk is about one idea, which keeps similarity scores sharp. The
corpus proved the point: the text PDFs produce exactly one 210–345 character chunk per
section (no arbitrary cuts, so overlap never fires), and the scan produces
paragraph-aligned chunks of ~560–690 characters. The values are settings
(`CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS`), but I would revisit them against a real
retrieval eval set before trusting them on longer documents.

**Scanned PDFs.** A page with fewer than 30 extractable characters is treated as a scan,
rasterised at 200 DPI and OCR'd with RapidOCR (pure pip, no Tesseract binary, so no
manual step on any OS). OCR returns lines, not paragraphs, so a line whose vertical gap
exceeds 2x the page's median line gap starts a new paragraph. That rule adapts to the
font instead of hard-coding one scan's spacing. I compared three OCR engines on the
sample: the older `rapidocr-onnxruntime` dropped spaces ("2026to"), EasyOCR misread a
digit ("1S percent", which is unacceptable for policy numbers), and `rapidocr` 3.x got
the page word- and digit-perfect.

## 5. RBAC approach

Identity comes only from the **verified JWT** (`sub`, `department`, `level`), never
from a chat message. From those claims `app/rag/rbac.py` builds a native Qdrant filter:

```
access_level <= user.level   AND   department IN {user.department, "hr"}
```

The "any user can read `hr`" exception is expressed as membership in a set, so it stays
one filter, and it is still bounded by `access_level` (a level-1 employee cannot read
`performance_review.pdf` just because it is HR).

The filter is sent **with the vector query** (`query_points(query_filter=...)`), so
Qdrant only considers permitted chunks. Restricted text never leaves the database, never
enters the LLM's context, and therefore cannot be leaked by prompt injection. Filtering
a top-k list in Python afterwards would fail both ways: restricted text would already be
in memory, and a user could get zero results even when permitted matches exist further
down the ranking. The RBAC fields are payload-indexed at ingestion.

`tests/test_rbac.py` runs the real filter against a real (in-memory) Qdrant and asserts
the exact set of visible files for each role. The agent and its retriever are built
**per connection** from the claims, so the model has no parameter through which to widen
its own access.

## 6. Async / parallel approach

`get_employee_context` starts all three lookups before awaiting any of them:

```python
profile, manager, team = await asyncio.gather(
    fetch_profile(uid), fetch_manager(uid), fetch_team(uid))
```

Each mock awaits `asyncio.sleep(1)`, which yields to the event loop, so the three
timers run concurrently and total time is the slowest call (~1 s), not the sum (~3 s).
Measured end to end: the tool-call event started at t=3.5 s and completed at t=4.5 s.
`tests/test_employee_context.py` asserts `1.0 <= elapsed < 1.5`. A whole-call timeout
(`asyncio.timeout`) stops one hung dependency from hanging the tool.

No blocking calls in `async def` paths: the tool uses `asyncio.sleep` (never
`time.sleep`), the embedding model runs in a thread executor (Agno's
`async_get_embedding`), and retrieval and the LLM use async clients. PDF parsing and OCR
happen only in the ingestion script, which is synchronous by design.

## 7. One thing I would improve

**Retrieval quality measurement and a reranker.** Right now relevance is controlled by
one similarity floor (`RETRIEVAL_MIN_SCORE=0.65`) that I tuned by eye on a handful of
questions: in-scope questions scored 0.74–0.82, personal/off-topic ones 0.40–0.62. With
more time I would build a small labelled question set, measure recall@k and answer
faithfulness, tune chunk size and the floor against it, and add a cross-encoder reranker
and BM25 hybrid search (exact terms like "PSU" or "USD 2,500" are where dense-only
retrieval is weakest).

## 8. One tradeoff I made

**Embedded Qdrant by default, instead of requiring Docker.** It makes the project run
with zero infrastructure, which is what a reviewer wants. The cost is that embedded
mode is single-process (ingest before serve, and no horizontal scaling) and its
"async" client is a thin wrapper over blocking calls; on 39 vectors that is
microseconds, but at scale it would block the event loop. Pointing `QDRANT_URL` at a
real server switches to true async HTTP with no code change (`docker-compose.yml` is
included).

---

## LLM provider: which and why

Provider choice is one setting (`LLM_PROVIDER`), resolved in `app/llm/factory.py` into an
Agno model object; nothing else in the code knows which provider is in use. Ollama,
Groq and OpenAI are wired up.

I default to **local Ollama with `qwen2.5`** because it needs no key or signup, so a
reviewer can run it immediately. **I recommend Groq (`llama-3.3-70b-versatile`) for
evaluation**: it streams much faster and follows instructions more reliably. I observed
real limits of the 7B local model on CPU: a cold first token of 10–30 s (model load),
occasional weak reasoning (one answer was correct but oddly phrased), and one
hallucinated figure before I set temperature to 0 and tightened the prompt. The
personal-question tool call worked on 4/4 phrasings after prompt tuning, but that is a
small sample.

## Design notes and assumptions

- **Agno usage.** Agno provides the agent loop, tool calling, streaming, session
  history, model adapters, the Qdrant wrapper and the embedder. Agno's built-in
  knowledge filters only support equality matches (I checked its source), which cannot
  express `access_level <= N` or the hr exception. So the RBAC filter is passed through
  Agno's `knowledge_retriever` hook as a native Qdrant `Filter`. Retrieval runs
  before every model call (`add_knowledge_to_context`), rather than as a tool the model
  might skip.
- **The tool takes no arguments.** The README specifies `get_employee_context(user_id)`.
  The underlying function has that signature, but the version exposed to the model is
  bound to the authenticated user's `sub`, so the model cannot ask about someone else.
- **HR exception vs access level.** I read "any user can access `hr` content" as "any
  department", still bounded by `access_level`. The alternative reading would let a
  level-1 employee see manager-only material.
- **No test user can see finance.** The three test users belong to `hr` and `exec`, and
  the stated rule is `department == user.department`, so `finance` documents are
  unreachable by them. That follows the spec literally; `test_rbac.py` shows a finance
  user can retrieve them.
- **Failure behaviour.** Bad, expired, malformed or missing tokens (and a 10 s auth
  timeout) get `auth_failed` and the socket closes (code 1008). An LLM, tool or
  retrieval failure sends `{"type":"error"}` followed by `{"type":"done"}` and keeps the
  connection open. `done` follows `error` so a client waiting for end-of-turn never
  hangs.
- **Known limitation.** If retrieval throws, Agno logs a warning and answers without
  references (the prompt tells the model to say it could not find the answer rather than
  guess). Fail-closed on access, but not a hard failure.
- **Not production yet.** HS256 with a shared secret is a mock; production would use an
  identity provider's asymmetric keys and re-check permissions on refresh. Conversation
  memory is in-process (lost on restart, not shared across instances), and there is no
  rate limiting or per-user concurrency cap.

## Layout

```
app/
  main.py               app factory + lifespan (loads embedder once)
  config.py  auth.py  schemas.py  tls.py
  api/ws_chat.py        protocol handler
  rag/rbac.py           the access-control filter
  rag/retriever.py      RBAC-filtered Qdrant search (Agno knowledge_retriever)
  rag/store.py          Agno Qdrant + embedder construction
  llm/factory.py        provider selection      llm/agent.py  Agno agent per connection
  tools/employee_context.py   the parallel tool
  ingestion/            extract (OCR fallback) · chunk · documents (access map) · pipeline
scripts/ingest.py  scripts/ws_demo.py
tests/
```
