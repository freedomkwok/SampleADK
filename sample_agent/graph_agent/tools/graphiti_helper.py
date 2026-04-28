"""Graphiti client wiring and async bridge for sync ADK tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from functools import lru_cache
from typing import Any, Coroutine, TypeVar

from graphiti_core.driver.oracle_pg_driver import OraclePGDriver
from graphiti_core.graphiti import Graphiti

T = TypeVar("T")


def run_graphiti_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run Graphiti async APIs from ADK sync tool functions (may be called under a running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


@lru_cache(maxsize=1)
def graphiti() -> Graphiti:
    """Singleton Graphiti over Oracle PG (``OraclePGDriver`` reads ``ORACLE_*`` env)."""
    driver = OraclePGDriver(
        graph_id=(os.getenv("GRAPHITI_GRAPH_ID") or os.getenv("GRAPH_ID") or "").strip() or None
    )
    return Graphiti(graph_driver=driver)


def compact_model(obj: Any) -> dict[str, Any]:
    """JSON-safe dict without large embedding fields."""
    if not hasattr(obj, "model_dump"):
        return {}
    return obj.model_dump(
        mode="json",
        exclude={
            "fact_embedding",
            "name_embedding",
            "summary_embedding",
            "content_embedding",
        },
    )
