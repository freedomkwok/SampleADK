"""ADK-exposed tools backed by the Graphiti Oracle PG client."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sample_agent.graph_agent.tools.graphiti_helper import GraphitiToolClient


_EMBEDDING_FIELDS = {
    "fact_embedding",
    "name_embedding",
    "summary_embedding",
    "content_embedding",
}


@lru_cache(maxsize=1)
def _client() -> GraphitiToolClient:
    return GraphitiToolClient()


def to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return _without_embeddings(dumped)
    if isinstance(value, dict):
        return _without_embeddings(dict(value))
    if hasattr(value, "__dict__"):
        return _without_embeddings(
            {
                key: raw_value
                for key, raw_value in vars(value).items()
                if not key.startswith("_")
            }
        )
    return {}


def _without_embeddings(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _EMBEDDING_FIELDS}


def extract_items(container: Any, *, preferred_keys: tuple[str, ...]) -> list[Any]:
    if isinstance(container, list):
        return list(container)
    payload = to_plain_dict(container)
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def trim_node_fields(raw_node: Any) -> dict[str, Any]:
    node_payload = to_plain_dict(raw_node)
    summary = node_payload.get("summary")
    score = node_payload.get("score")
    uuid_ = node_payload.get("uuid_") or node_payload.get("uuid")
    return {
        "name": node_payload.get("name"),
        "attributes": node_payload.get("attributes")
        if isinstance(node_payload.get("attributes"), dict)
        else {},
        "metadata": node_payload.get("metadata")
        if isinstance(node_payload.get("metadata"), dict)
        else {},
        "summary": summary if isinstance(summary, str) else "",
        "score": score if isinstance(score, (int, float)) else None,
        "uuid_": uuid_ if isinstance(uuid_, str) else "",
    }


def search_nodes(query: str, limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Find graph entities relevant to a natural-language query."""
    client = _client()
    resolved_graph_id = client.resolve_graph_id(graph_id)
    normalized_query = query.strip()
    if not normalized_query:
        return {"graph_id": resolved_graph_id, "nodes": [], "count": 0}
    response = client.client.graph.search(
        query=normalized_query,
        graph_id=resolved_graph_id or None,
        scope="nodes",
        limit=max(1, limit),
    )
    nodes = [trim_node_fields(node) for node in (getattr(response, "nodes", None) or []) if node]
    return {"graph_id": resolved_graph_id, "nodes": nodes, "count": len(nodes)}


def search_edges(query: str, limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Find relationship facts relevant to a natural-language query."""
    client = _client()
    resolved_graph_id = client.resolve_graph_id(graph_id)
    normalized_query = query.strip()
    if not normalized_query:
        return {"graph_id": resolved_graph_id, "edges": [], "count": 0}
    response = client.client.graph.search(
        query=normalized_query,
        graph_id=resolved_graph_id or None,
        scope="edges",
        limit=max(1, limit),
    )
    edges = [to_plain_dict(edge) for edge in (getattr(response, "edges", None) or []) if edge]
    return {"graph_id": resolved_graph_id, "edges": edges, "count": len(edges)}


def get_edges_for_node(node_uuid: str) -> dict[str, Any]:
    """Fetch all edges directly connected to one node."""
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "edges": [], "count": 0}
    response = client.client.graph.node.get_edges(node_uuid=normalized_uuid)
    edges = [to_plain_dict(edge) for edge in extract_items(response, preferred_keys=("edges",)) if edge]
    return {"node_uuid": normalized_uuid, "edges": edges, "count": len(edges)}


def get_node_by_id(node_uuid: str) -> dict[str, Any]:
    """Fetch one node by UUID and return a compact, model-safe shape."""
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "node": None}
    try:
        node = client.client.graph.node.get(uuid_=normalized_uuid)
    except Exception as exc:  # noqa: BLE001
        return {"node_uuid": normalized_uuid, "node": None, "error": str(exc)}
    return {"node_uuid": normalized_uuid, "node": trim_node_fields(node)}


def search_around_node(
    node_uuid: str,
    query: str = "",
    limit: int = 10,
    graph_id: str = "",
) -> dict[str, Any]:
    """Build a neighborhood context bundle around a node."""
    node_result = get_node_by_id(node_uuid=node_uuid)
    node = node_result.get("node")
    if not isinstance(node, dict):
        return {
            "node_uuid": str(node_uuid).strip(),
            "node": None,
            "edges": [],
            "related_nodes": [],
            "related_edges": [],
        }
    edge_result = get_edges_for_node(node_uuid=node_uuid)
    fallback_query = query.strip() or str(node.get("name") or node.get("summary") or node_uuid)
    related_nodes = search_nodes(query=fallback_query, limit=limit, graph_id=graph_id)
    related_edges = search_edges(query=fallback_query, limit=limit, graph_id=graph_id)
    return {
        "node_uuid": str(node_uuid).strip(),
        "node": node,
        "edges": edge_result.get("edges", []),
        "related_nodes": related_nodes.get("nodes", []),
        "related_edges": related_edges.get("edges", []),
    }
