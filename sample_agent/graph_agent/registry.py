"""Registration helpers for local Graphiti graph_agent (zep_agent-style)."""

from __future__ import annotations

from pathlib import Path

from vertexai.preview.reasoning_engines import A2aAgent

from sample_agent.a2a import ConfiguredA2aExecutor, build_agent_card_from_yaml
from sample_agent.graph_agent._env import bootstrap_env

bootstrap_env()
config_path = Path(__file__).with_name("config.yaml")
agent_card = build_agent_card_from_yaml(config_path, config_section="card_config")


def build_local_a2a_graph_agent() -> A2aAgent:
    """Build local A2A agent from YAML-driven executor."""
    agent = A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=lambda: ConfiguredA2aExecutor(
            config_path=config_path, config_section="executor_config"
        ),
    )
    agent.set_up()
    return agent
