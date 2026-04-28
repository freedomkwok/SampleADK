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


def _env_str(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return value or None


def _env_int(name: str) -> int | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _env_bool(name: str) -> bool | None:
    raw = _env_str(name)
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _oracle_driver_kwargs() -> dict[str, Any]:
    graph_id = _env_str("GRAPHITI_GRAPH_ID") or _env_str("GRAPH_ID")
    dsn = _env_str("ORACLE_DSN")
    uri = None if dsn else _env_str("ORACLE_URI")
    user = _env_str("ORACLE_USER")
    password = _env_str("ORACLE_PASSWORD")
    max_coroutines = _env_int("ORACLE_MAX_COROUTINES")
    log_queries = _env_bool("ORACLE_LOG_QUERIES")

    connect_kwargs: dict[str, int] = {}
    for env_name, key in (
        ("ORACLE_POOL_MIN", "min"),
        ("ORACLE_POOL_MAX", "max"),
        ("ORACLE_POOL_INCREMENT", "increment"),
    ):
        value = _env_int(env_name)
        if value is not None:
            connect_kwargs[key] = value

    if dsn is None and uri is None:
        raise ValueError("Oracle Graphiti mode requires ORACLE_DSN (preferred) or ORACLE_URI.")

    kwargs: dict[str, Any] = {"graph_id": graph_id, "dsn": dsn, "uri": uri}
    if user is not None:
        kwargs["user"] = user
    if password is not None:
        kwargs["password"] = password
    if max_coroutines is not None:
        kwargs["max_coroutines"] = max_coroutines
    if log_queries is not None:
        kwargs["log_queries"] = log_queries
    if connect_kwargs:
        kwargs["connect_kwargs"] = connect_kwargs
    return kwargs


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
    """Singleton Graphiti over Oracle PG (DSN-first, URI fallback)."""
    driver = OraclePGDriver(**_oracle_driver_kwargs())
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
