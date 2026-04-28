"""ADK-exposed tools backed by Graphiti hybrid search (Oracle PG driver)."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import Context
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from sample_agent.graph_agent.tools.graphiti_helper import compact_model, graphiti, run_graphiti_coroutine


def _resolved_group_ids(group_id: str, tool_context: Context) -> list[str] | None:
    gid = (group_id or "").strip()
    if not gid:
        gid = str(tool_context.state.get("graph_id") or "").strip()
    if not gid:
        gid = (os.getenv("GRAPHITI_DEFAULT_GROUP_ID") or "").strip()
    return [gid] if gid else None


def _normalized_query(query: str) -> str:
    return query.strip()


async def _search_facts_async(query: str, limit: int, group_ids: list[str] | None) -> dict[str, Any]:
    client = graphiti()
    normalized_query = _normalized_query(query)
    edges = await client.search(query=normalized_query, num_results=max(1, limit), group_ids=group_ids)
    return {
        "query": normalized_query,
        "group_ids": group_ids,
        "edges": [compact_model(edge) for edge in edges],
        "count": len(edges),
    }


async def _hybrid_search_async(query: str, limit: int, group_ids: list[str] | None) -> dict[str, Any]:
    client = graphiti()
    cfg = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": max(1, limit)})
    normalized_query = _normalized_query(query)
    results = await client.search_(query=normalized_query, config=cfg, group_ids=group_ids)
    return {
        "query": normalized_query,
        "group_ids": group_ids,
        "edges": [compact_model(edge) for edge in results.edges],
        "nodes": [compact_model(node) for node in results.nodes],
        "episodes": [compact_model(ep) for ep in results.episodes],
        "communities": [compact_model(c) for c in results.communities],
        "counts": {
            "edges": len(results.edges),
            "nodes": len(results.nodes),
            "episodes": len(results.episodes),
            "communities": len(results.communities),
        },
    }


def graph_search_facts(
    query: str,
    limit: int = 10,
    group_id: str = "",
    *,
    tool_context: Context,
) -> dict[str, Any]:
    """Hybrid edge search returning ranked entity-edge facts for a natural-language query.

    Use ``group_id`` to scope to one Graphiti partition, or rely on session ``graph_id``
    (from A2A message metadata) / ``GRAPHITI_DEFAULT_GROUP_ID``.
    """
    group_ids = _resolved_group_ids(group_id, tool_context)
    normalized_query = _normalized_query(query)
    if not normalized_query:
        return {"query": "", "group_ids": group_ids, "edges": [], "count": 0}
    try:
        return run_graphiti_coroutine(_search_facts_async(normalized_query, limit, group_ids))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "query": normalized_query, "group_ids": group_ids}


def graph_hybrid_search(
    query: str,
    limit: int = 10,
    group_id: str = "",
    *,
    tool_context: Context,
) -> dict[str, Any]:
    """Full hybrid search across edges, nodes, episodes, and communities (RRF recipe)."""
    group_ids = _resolved_group_ids(group_id, tool_context)
    normalized_query = _normalized_query(query)
    if not normalized_query:
        return {
            "query": "",
            "group_ids": group_ids,
            "edges": [],
            "nodes": [],
            "episodes": [],
            "communities": [],
            "counts": {"edges": 0, "nodes": 0, "episodes": 0, "communities": 0},
        }
    try:
        return run_graphiti_coroutine(_hybrid_search_async(normalized_query, limit, group_ids))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "query": normalized_query, "group_ids": group_ids}
