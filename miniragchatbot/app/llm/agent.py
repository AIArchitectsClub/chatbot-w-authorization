"""The Agno agent: one per WebSocket connection, bound to one authenticated user."""

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.vectordb.qdrant import Qdrant

from app.config import Settings
from app.llm.factory import make_model
from app.rag.retriever import make_rbac_retriever
from app.schemas import UserClaims
from app.tools.employee_context import make_employee_context_tool

HISTORY_TURNS = 5

INSTRUCTIONS = [
    "You are an internal assistant that answers questions about Acme Corp company policy.",
    "Answer policy questions ONLY from the <references> provided with the question. "
    "Quote figures (days, amounts, percentages) exactly as they appear there, and "
    "never add a number, rate or rule that is not written in the references.",
    "If the references do not contain the answer, say you could not find it in the "
    "documents you have access to. Do not guess, do not name documents you were not "
    "shown, and do not say whether other documents exist.",
    "If the question mentions the user's own situation (words like 'my', 'me', 'I', "
    "e.g. 'my grade', 'who is my manager', 'how much leave do I have given my grade'), "
    "you MUST call get_employee_context BEFORE answering, even when policy references "
    "are also present. Then combine the tool result with the references, for example "
    "to check whether a rule depends on grade. Never invent personal details.",
    "Keep answers concise and plain text. Cite a document name only for facts that "
    "come from the references; never cite a source for tool results.",
]


def build_agent(user: UserClaims, vector_db: Qdrant, settings: Settings) -> Agent:
    """Create the agent for one connection.

    Everything user-specific is bound here from the verified JWT claims:
      * the retriever (RBAC filter),
      * the tool (whose data it fetches),
      * the conversation history (a private in-memory store).
    """
    return Agent(
        model=make_model(settings),
        instructions=INSTRUCTIONS,
        tools=[make_employee_context_tool(user.sub, settings.tool_timeout_seconds)],
        # RAG on every turn: retrieve first (RBAC-filtered), then let the model answer.
        knowledge_retriever=make_rbac_retriever(
            user, vector_db, settings.retrieval_top_k, settings.retrieval_min_score
        ),
        add_knowledge_to_context=True,
        search_knowledge=False,
        db=InMemoryDb(),
        add_history_to_context=True,
        num_history_runs=HISTORY_TURNS,
        markdown=False,
    )
