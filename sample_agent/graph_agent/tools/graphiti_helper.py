"""Graphiti client wiring and async bridge for sync ADK tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from functools import lru_cache
from typing import Any, Coroutine, TypeVar

from graphiti_client import GraphitiOraclePGClient, GraphitiOraclePGConnection  # type: ignore[import-not-found]
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

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


def _oracle_connection() -> GraphitiOraclePGConnection:
    graph_id = _env_str("GRAPHITI_GRAPH_ID") or _env_str("GRAPH_ID")
    dsn = _env_str("ORACLE_DSN")
    user = _env_str("ORACLE_USER")
    password = _env_str("ORACLE_PASSWORD")
    max_coroutines = _env_int("ORACLE_MAX_COROUTINES")

    connect_kwargs: dict[str, int] = {}
    for env_name, key in (
        ("ORACLE_POOL_MIN", "min"),
        ("ORACLE_POOL_MAX", "max"),
        ("ORACLE_POOL_INCREMENT", "increment"),
    ):
        value = _env_int(env_name)
        if value is not None:
            connect_kwargs[key] = value

    if not dsn or not user or not password:
        raise ValueError("Oracle Graphiti mode requires ORACLE_DSN, ORACLE_USER, and ORACLE_PASSWORD.")
    if not graph_id:
        raise ValueError("Oracle Graphiti mode requires GRAPHITI_GRAPH_ID or GRAPH_ID.")

    return GraphitiOraclePGConnection(
        dsn=dsn,
        user=user,
        password=password,
        graph_id=graph_id,
        connect_kwargs=connect_kwargs or None,
        max_coroutines=max_coroutines,
        log_queries=bool(_env_bool("ORACLE_LOG_QUERIES")),
    )


def _build_llm_client() -> OpenAIGenericClient:
    api_key = _env_str("LLM_API_KEY") or _env_str("OPENAI_API_KEY")
    base_url = _env_str("LLM_BASE_URL") or _env_str("OPENAI_BASE_URL")
    model = _env_str("GRAPHITI_LLM_MODEL") or _env_str("LLM_MODEL_NAME")
    small_model = _env_str("GRAPHITI_LLM_SMALL_MODEL")
    temperature = float(_env_str("GRAPHITI_LLM_TEMPERATURE") or "0")
    max_tokens = int(_env_str("GRAPHITI_LLM_MAX_TOKENS") or "50000")

    config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        small_model=small_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return OpenAIGenericClient(config=config, max_tokens=max_tokens)


def _build_embedder() -> OpenAIEmbedder:
    api_key = _env_str("GRAPHITI_EMBEDDING_API_KEY") or _env_str("OPENAI_API_KEY")
    base_url = _env_str("GRAPHITI_EMBEDDING_BASE_URL") or _env_str("OPENAI_BASE_URL")
    embedding_model = _env_str("GRAPHITI_EMBEDDING_MODEL") or _env_str("OPENAI_EMBEDDING_MODEL")

    if embedding_model:
        config = OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
        )
    else:
        config = OpenAIEmbedderConfig(api_key=api_key, base_url=base_url)
    return OpenAIEmbedder(config=config)


def _embedder_max_batch_size() -> int | None:
    return _env_int("GRAPHITI_EMBEDDING_MAX_BATCH_SIZE")


def run_graphiti_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run Graphiti async APIs from ADK sync tool functions (may be called under a running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


@lru_cache(maxsize=1)
def graphiti_client() -> GraphitiOraclePGClient:
    """Singleton Graphiti Oracle PG client with explicit LLM/embedder setup."""
    connection = _oracle_connection()
    return GraphitiOraclePGClient.from_connection(
        dsn=connection.dsn,
        user=connection.user,
        password=connection.password,
        graph_id=connection.graph_id,
        llm_client=_build_llm_client(),
        embedder=_build_embedder(),
        connect_kwargs=connection.connect_kwargs,
        max_coroutines=connection.max_coroutines,
        log_queries=connection.log_queries,
        embedder_max_batch_size=_embedder_max_batch_size(),
        run_async=run_graphiti_coroutine,
    )


class GraphitiToolClient:
    """Small adapter matching the zep_agent tool client shape."""

    def __init__(self, client: GraphitiOraclePGClient | None = None) -> None:
        self.client = client or graphiti_client()

    def resolve_graph_id(self, graph_id: str = "") -> str:
        return (
            str(graph_id or "").strip()
            or _env_str("GRAPHITI_DEFAULT_GROUP_ID")
            or _env_str("GRAPHITI_GRAPH_ID")
            or _env_str("GRAPH_ID")
            or ""
        )


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
