"""ADK LlmAgent for Graphiti-backed retrieval (zep_agent-style, no agent_core)."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import LlmAgent

from sample_agent.graph_agent._env import bootstrap_env
from sample_agent.graph_agent.tools import (
    get_edges_for_node,
    get_node_by_id,
    search_around_node,
    search_edges,
    search_nodes,
)

bootstrap_env()

_AGENT_NAME = "graph_retrieval_agent"
_DEFAULT_INSTRUCTION = (
    "You are a graph retrieval assistant backed by Graphiti (Oracle PG). Use search_nodes "
    "to find relevant entities, search_edges for relationship facts, get_node_by_id when "
    "you already have a node UUID, get_edges_for_node to expand a known node, and "
    "search_around_node for neighborhood context. Always cite which tool results support "
    "your answer. If graph_id is unknown, rely on the configured default graph."
)


def _is_enabled(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def build_graph_llm_agent(
    *,
    langfuse_client: Any = None,
    instruction_prompt_name: str | None = None,
    instruction_prompt_label: str | None = None,
) -> LlmAgent:
    """Build the LLM tool-calling agent (YAML ``adk_agent_builder`` target).

    Default: Gemini via ``ADK_MODEL``. Set ``ADK_USE_LITELLM_CHATGPT=1`` to use
    OpenAI through ``ChatGPTClient`` (``LiteLlm`` + ``register_chatgpt_litellm``);
    requires ``OPENAI_API_KEY`` and uses ``OPENAI_MODEL`` for the OpenAI model id.
    """
    del langfuse_client, instruction_prompt_name, instruction_prompt_label
    use_litellm = _is_enabled(os.getenv("ADK_USE_LITELLM_CHATGPT", ""))
    if use_litellm:
        from sample_agent.litellm_chatgpt_adapter import build_adk_litellm_for_chatgpt

        model = build_adk_litellm_for_chatgpt()
    else:
        model = (os.getenv("ADK_MODEL") or "gemini-2.0-flash").strip()
    return LlmAgent(
        model=model,
        name=_AGENT_NAME,
        description="Graphiti hybrid search and fact retrieval over Oracle PG.",
        instruction=_DEFAULT_INSTRUCTION,
        tools=[
            search_nodes,
            get_edges_for_node,
            search_edges,
            get_node_by_id,
            search_around_node,
        ],
    )
