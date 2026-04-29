"""ADK-exposed tools backed by the Graphiti Oracle PG client."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from graphiti_core.search.search_filters import SearchFilters
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


def _graphiti_search_filter(raw_filter: dict[str, Any] | None) -> SearchFilters | None:
    if not raw_filter:
        return None
    return SearchFilters(**raw_filter)


def _graph_result_score(item: dict[str, Any]) -> float:
    score = item.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return float("-inf")
    return float(score)


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


def search_nodes(
    query: str,
    limit: int = 10,
    graph_id: str = "",
    search_filter: dict[str, Any] | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    center_node_uuid: str = "",
) -> dict[str, Any]:
    """Find Graphiti entity nodes relevant to a natural-language query.

    Best for entity discovery when the model needs candidate people, objects,
    concepts, skills, or other graph entities before taking deeper actions.

    Useful filters:
      node_labels: include only nodes with these labels.
      edge_types, edge_uuids: mainly for related edge/fact constraints.
      property_filters: check stored properties by property_name,
        comparison_operator, and property_value. Operators include =, <>, >,
        <, >=, <=, IS NULL, and IS NOT NULL.
      created_at, valid_at, invalid_at, expired_at: 2D date filter arrays.
        Inner lists are AND; outer lists are OR.

    Search tuning priority:
      High: search_filter. Use for explicit checks like label, type, date, or
        property constraints.
      High: bfs_origin_node_uuids. Passing this list seeds Graphiti BFS search
        from known entity or episodic node UUIDs. Example ["node-123", "node-456"].
      Medium: center_node_uuid. Use when results should stay close to a known
        entity node in the graph. Example "node-123".

    Args:
        query: Natural-language search text.
        limit: Maximum number of node hits to return (minimum 1).
        graph_id: Target Graphiti group/graph ID; falls back to configured default when empty.
        search_filter: Optional Graphiti SearchFilters dict. Examples:
          {"node_labels": ["Person"]}
          {"property_filters": [{"property_name": "status",
            "comparison_operator": "=", "property_value": "active"}]}
          {"created_at": [[{"comparison_operator": ">=",
            "date": "2026-01-01T00:00:00Z"}]]}
        bfs_origin_node_uuids: Optional origin node UUIDs for Graphiti BFS search.
            Supplying this parameter alone is enough to seed BFS.
        center_node_uuid: Optional node UUID for distance-aware reranking.

    Returns:
        {
          "graph_id": str,
          "nodes": [  # sorted by score descending
            {"uuid_": str, "name": str | None, "attributes": dict, "metadata": dict, "summary": str, "score": float | None}
          ],
          "count": int
        }
    """
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
        search_filter=_graphiti_search_filter(search_filter),
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        center_node_uuid=center_node_uuid.strip() or None,
    )
    nodes = sorted(
        [trim_node_fields(node) for node in (getattr(response, "nodes", None) or []) if node],
        key=_graph_result_score,
        reverse=True,
    )
    return {"graph_id": resolved_graph_id, "nodes": nodes, "count": len(nodes)}


def search_edges(
    query: str,
    limit: int = 10,
    graph_id: str = "",
    search_filter: dict[str, Any] | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    center_node_uuid: str = "",
) -> dict[str, Any]:
    """Find Graphiti relationship facts relevant to a natural-language query.

    Best when the model needs relation/evidence-level context: facts, links,
    provenance-bearing relationships, and temporal validity.

    Useful filters:
      edge_types: include only these relationship/fact types.
      edge_uuids: restrict to known edge UUIDs.
      node_labels: constrain labels on attached entity nodes.
      property_filters: check stored properties by property_name,
        comparison_operator, and property_value. Operators include =, <>, >,
        <, >=, <=, IS NULL, and IS NOT NULL.
      created_at, valid_at, invalid_at, expired_at: 2D date filter arrays.
        Inner lists are AND; outer lists are OR.

    Search tuning priority:
      High: search_filter. Use for explicit checks like relationship type,
        edge UUID, date validity, or property constraints.
      High: bfs_origin_node_uuids. Passing this list seeds Graphiti BFS search
        from known entity or episodic node UUIDs. Example ["node-123", "node-456"].
      Medium: center_node_uuid. Use when facts should stay close to a known
        entity node in the graph. Example "node-123".

    Args:
        query: Natural-language search text.
        limit: Maximum number of edge hits to return (minimum 1).
        graph_id: Target Graphiti group/graph ID; falls back to configured default when empty.
        search_filter: Optional Graphiti SearchFilters dict. Examples:
          {"edge_types": ["WORKS_AT"]}
          {"edge_uuids": ["edge-123"]}
          {"property_filters": [{"property_name": "confidence",
            "comparison_operator": ">=", "property_value": 0.8}]}
          {"valid_at": [[{"comparison_operator": "<=",
            "date": "2026-04-01T00:00:00Z"}]]}
        bfs_origin_node_uuids: Optional origin node UUIDs for Graphiti BFS search.
            Supplying this parameter alone is enough to seed BFS.
        center_node_uuid: Optional node UUID for distance-aware reranking.

    Returns:
        {
          "graph_id": str,
          "edges": [edge_dict, ...],  # sorted by score descending
          "count": int
        }
    """
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
        search_filter=_graphiti_search_filter(search_filter),
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        center_node_uuid=center_node_uuid.strip() or None,
    )
    edges = sorted(
        [to_plain_dict(edge) for edge in (getattr(response, "edges", None) or []) if edge],
        key=_graph_result_score,
        reverse=True,
    )
    return {"graph_id": resolved_graph_id, "edges": edges, "count": len(edges)}


def get_edges_for_node(node_uuid: str) -> dict[str, Any]:
    """Fetch all relationship facts directly connected to one node.

    Best for local graph expansion after search_nodes returns a useful entity
    UUID. Use this when the model already has a node identifier and needs the
    surrounding facts without running another semantic search.

    Args:
        node_uuid: UUID of the anchor node.

    Returns:
        {
          "node_uuid": str,
          "edges": [edge_dict, ...],  # sorted by score descending when scores exist
          "count": int
        }
    """
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "edges": [], "count": 0}
    response = client.client.graph.node.get_edges(node_uuid=normalized_uuid)
    edges = sorted(
        [to_plain_dict(edge) for edge in extract_items(response, preferred_keys=("edges",)) if edge],
        key=_graph_result_score,
        reverse=True,
    )
    return {"node_uuid": normalized_uuid, "edges": edges, "count": len(edges)}


def get_node_by_id(node_uuid: str) -> dict[str, Any]:
    """Fetch one Graphiti node by UUID and return a compact shape.

    Best when the model already has a node UUID and needs stable identity,
    attributes, metadata, summary, or score before deciding the next graph step.

    Args:
        node_uuid: UUID of the node to retrieve.

    Returns:
        {
          "node_uuid": str,
          "node": {"uuid_", "name", "attributes", "metadata", "summary", "score"} | None,
          "error": str (optional)
        }
    """
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
    """Build a neighborhood context bundle around a known node.

    Combines direct node lookup, connected edges, and related node/edge search
    into one response for routing or reasoning steps. Use this when the model
    has a strong anchor node and wants a compact local context bundle.

    Args:
        node_uuid: Anchor node UUID.
        query: Optional related-search query. If empty, uses the node name,
            summary, or UUID.
        limit: Maximum results for related node and edge searches.
        graph_id: Target Graphiti group/graph ID for related searches.

    Returns:
        {
          "node_uuid": str,
          "node": compact_node | None,
          "edges": [...],
          "related_nodes": [compact_node, ...],
          "related_edges": [...]
        }
    """
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
