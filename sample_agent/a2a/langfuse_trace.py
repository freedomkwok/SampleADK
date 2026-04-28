"""Minimal Langfuse async chain wrapper (inlined from llm_inference_core pattern)."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_with_langfuse_trace(
    *,
    langfuse: Any,
    langfuse_type: str,
    trace_name: str,
    trace_input: Any,
    runner: Callable[[], Awaitable[Any]],
    on_success: Callable[[Any, Any], None] | None = None,
    on_observation_start: Callable[[Any], None] | None = None,
    metadata: dict[str, Any] | None = None,
    model_name: str | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    flush_on_exit: bool = False,
) -> Any:
    """Run an async operation wrapped in a Langfuse observation."""
    if langfuse is None:
        return await runner()

    logger.info("running trace=%s", trace_name)

    obs_kwargs: dict[str, Any] = {
        "name": trace_name,
        "as_type": langfuse_type,
        "model": model_name,
        "input": trace_input,
        "metadata": metadata or {},
    }
    if trace_id or parent_span_id:
        trace_context: dict[str, str] = {}
        if trace_id:
            trace_context["trace_id"] = trace_id
        if parent_span_id:
            trace_context["parent_span_id"] = parent_span_id
        obs_kwargs["trace_context"] = trace_context

    with langfuse.start_as_current_observation(**obs_kwargs) as observation:
        if on_observation_start is not None:
            on_observation_start(observation)
        try:
            result = await runner()
            if on_success is not None:
                try:
                    on_success(result, observation)
                except Exception:
                    logger.debug("on_success callback failed", exc_info=True)
            return result
        except Exception as exc:
            try:
                observation.update(output={"error": str(exc)})
            except Exception:
                logger.debug("failed to update observation error output", exc_info=True)
            raise
        finally:
            if flush_on_exit:
                try:
                    langfuse.flush()
                except Exception:
                    logger.debug("langfuse flush failed", exc_info=True)
