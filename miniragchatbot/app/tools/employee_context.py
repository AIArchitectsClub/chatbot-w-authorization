"""The `get_employee_context` tool: three slow lookups run concurrently."""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Each mock stands in for a real network call; asyncio.sleep yields to the
# event loop instead of blocking it (time.sleep would serialise everything).
MOCK_LATENCY_SECONDS = 1.0


async def fetch_profile(user_id: str) -> dict[str, Any]:
    await asyncio.sleep(MOCK_LATENCY_SECONDS)
    return {"name": "Jane Doe", "grade": "Senior"}


async def fetch_manager(user_id: str) -> dict[str, Any]:
    await asyncio.sleep(MOCK_LATENCY_SECONDS)
    return {"manager": "John Smith"}


async def fetch_team(user_id: str) -> dict[str, Any]:
    await asyncio.sleep(MOCK_LATENCY_SECONDS)
    return {"team_size": 8, "team_name": "Platform"}


async def get_employee_context(user_id: str, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch profile, manager and team info concurrently (~1s, not ~3s).

    All three coroutines are started before any is awaited; `gather` then waits
    once, so total time is the slowest call rather than the sum.
    """
    started = time.perf_counter()
    async with asyncio.timeout(timeout):
        profile, manager, team = await asyncio.gather(
            fetch_profile(user_id), fetch_manager(user_id), fetch_team(user_id)
        )
    logger.info(
        "get_employee_context user=%s elapsed=%.2fs", user_id, time.perf_counter() - started
    )
    return {"user_id": user_id, **profile, **manager, **team}


def make_employee_context_tool(
    user_id: str, timeout: float
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Build the agent-facing tool, bound to the authenticated user.

    The model cannot choose whose data to fetch: `user_id` comes from the
    verified JWT, not from model output.
    """

    async def get_employee_context_for_current_user() -> dict[str, Any]:
        """Look up the current user's own details: name, grade, manager, team name and team size.

        ALWAYS call this when the question is about the user personally, i.e. it
        contains "my", "me" or "I": "who is my manager", "what is my grade",
        "how big is my team", "how much leave do I have given my grade".
        Takes no arguments; the user is already known.
        """
        return await get_employee_context(user_id, timeout)

    get_employee_context_for_current_user.__name__ = "get_employee_context"
    return get_employee_context_for_current_user
