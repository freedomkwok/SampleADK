"""Debug runner: local A2A + ADK flow for graph_agent."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sample_agent.a2a import OrchestrationMode, run_local_a2a_orchestration
from sample_agent.graph_agent.registry import build_local_a2a_graph_agent


async def _main() -> None:
    mode = OrchestrationMode.HOST_DRIVEN
    a2a_agent = build_local_a2a_graph_agent()
    metadata: dict[str, str] = {}
    gid = os.getenv("GRAPHITI_DEFAULT_GROUP_ID", "").strip()
    if gid:
        metadata["graph_id"] = gid
    result = await run_local_a2a_orchestration(
        a2a_agent=a2a_agent,
        message_text="What does the graph contain that is relevant to the user's domain?",
        mode=mode,
        metadata=metadata or None,
    )
    print("task_id:", result.task_id)
    print("task_status:", result.task_status)
    print("final_text:\n", result.final_text)


if __name__ == "__main__":
    asyncio.run(_main())
