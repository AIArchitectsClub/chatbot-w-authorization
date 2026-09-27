import asyncio
import time

import pytest

from app.tools import employee_context
from app.tools.employee_context import get_employee_context, make_employee_context_tool


async def test_three_lookups_run_in_parallel():
    started = time.perf_counter()
    result = await get_employee_context("emp-001")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5, f"expected ~1s (parallel), took {elapsed:.2f}s (serial would be ~3s)"
    assert elapsed >= 1.0
    assert result["name"] == "Jane Doe"
    assert result["manager"] == "John Smith"
    assert result["team_name"] == "Platform" and result["team_size"] == 8


async def test_tool_is_bound_to_authenticated_user_and_takes_no_arguments():
    tool = make_employee_context_tool("mgr-002", timeout=5.0)
    assert tool.__name__ == "get_employee_context"
    result = await tool()
    assert result["user_id"] == "mgr-002"


async def test_slow_dependency_times_out(monkeypatch):
    async def slow(_user_id):
        await asyncio.sleep(5)

    monkeypatch.setattr(employee_context, "fetch_team", slow)
    with pytest.raises(TimeoutError):
        await get_employee_context("emp-001", timeout=0.2)
