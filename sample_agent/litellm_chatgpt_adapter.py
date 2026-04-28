"""Register a LiteLLM custom provider that forwards chat completions to ``ChatGPTClient``."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.utils import custom_llm_setup
from openai import OpenAI

from sample_agent.chatgpt_client import ChatGPTClient

CHATGPT_LITELLM_PROVIDER = "sample_agent_chatgpt"

_OPENAI_CHAT_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "n",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "tools",
        "tool_choice",
        "response_format",
        "seed",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "reasoning_effort",
        "verbosity",
        "extra_headers",
        "extra_query",
        "extra_body",
    }
)


def default_litellm_model_id(chatgpt: ChatGPTClient | None = None) -> str:
    """Model string for ``litellm.acompletion`` / ADK ``LiteLlm`` (``provider/openai-model-id``)."""
    cg = chatgpt or ChatGPTClient()
    return f"{CHATGPT_LITELLM_PROVIDER}/{cg.model}"


def _openai_kwargs(optional_params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in optional_params.items() if k in _OPENAI_CHAT_KEYS}


def _openai_model_id(litellm_model: str, chatgpt: ChatGPTClient) -> str:
    raw = (litellm_model or "").strip()
    if not raw:
        return chatgpt.model
    if "/" in raw:
        suffix = raw.split("/", 1)[1].strip()
        return suffix or chatgpt.model
    return raw


class ChatGPTLiteLLMAdapter(CustomLLM):
    """LiteLLM ``CustomLLM`` that calls OpenAI via in-repo ``ChatGPTClient`` (``AsyncOpenAI``)."""

    def __init__(self, chatgpt: ChatGPTClient | None = None) -> None:
        super().__init__()
        self._chatgpt = chatgpt or ChatGPTClient()

    def _sync_client(self) -> OpenAI:
        return OpenAI(
            api_key=self._chatgpt.client.api_key,
            base_url=str(self._chatgpt.client.base_url)
            if self._chatgpt.client.base_url
            else None,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> ModelResponse:
        del (
            api_base,
            custom_prompt_dict,
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            acompletion,
            litellm_params,
            logger_fn,
            client,
            headers,
        )
        openai_model = _openai_model_id(model, self._chatgpt)
        kwargs = _openai_kwargs(optional_params)
        raw = self._sync_client().chat.completions.create(
            model=openai_model,
            messages=messages,
            timeout=timeout,
            **kwargs,
        )
        return convert_to_model_response_object(
            response_object=raw.model_dump(),
            model_response_object=model_response,
            response_type="completion",
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> ModelResponse:
        del (
            api_base,
            custom_prompt_dict,
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            acompletion,
            litellm_params,
            logger_fn,
            client,
            headers,
        )
        openai_model = _openai_model_id(model, self._chatgpt)
        kwargs = _openai_kwargs(optional_params)
        raw = await self._chatgpt.client.chat.completions.create(
            model=openai_model,
            messages=messages,
            timeout=timeout,
            **kwargs,
        )
        return convert_to_model_response_object(
            response_object=raw.model_dump(),
            model_response_object=model_response,
            response_type="completion",
        )

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        del (
            api_base,
            custom_prompt_dict,
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            acompletion,
            litellm_params,
            logger_fn,
            client,
            headers,
        )
        openai_model = _openai_model_id(model, self._chatgpt)
        kwargs = _openai_kwargs(optional_params)
        kwargs["stream_options"] = {"include_usage": True}
        stream = self._sync_client().chat.completions.create(
            model=openai_model,
            messages=messages,
            stream=True,
            timeout=timeout,
            **kwargs,
        )
        yield from _iter_openai_stream_chunks(stream)

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        del (
            api_base,
            custom_prompt_dict,
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            acompletion,
            litellm_params,
            logger_fn,
            client,
            headers,
        )
        openai_model = _openai_model_id(model, self._chatgpt)
        kwargs = _openai_kwargs(optional_params)
        kwargs["stream_options"] = {"include_usage": True}
        stream = await self._chatgpt.client.chat.completions.create(
            model=openai_model,
            messages=messages,
            stream=True,
            timeout=timeout,
            **kwargs,
        )
        async for item in _aiter_openai_stream_chunks(stream):
            yield item


def _usage_block(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return None


def _iter_openai_stream_chunks(stream: Any) -> Iterator[GenericStreamingChunk]:
    for event in stream:
        yield from _openai_chunk_to_generic_chunks(event)


async def _aiter_openai_stream_chunks(stream: Any) -> AsyncIterator[GenericStreamingChunk]:
    async for event in stream:
        for chunk in _openai_chunk_to_generic_chunks(event):
            yield chunk


def _openai_chunk_to_generic_chunks(event: Any) -> list[GenericStreamingChunk]:
    usage = _usage_block(getattr(event, "usage", None))
    choices = getattr(event, "choices", None) or []
    if not choices:
        if usage:
            return [
                {
                    "text": "",
                    "is_finished": True,
                    "finish_reason": "stop",
                    "usage": usage,
                }
            ]
        return []

    out: list[GenericStreamingChunk] = []
    for ch in choices:
        delta = ch.delta
        finish = ch.finish_reason
        text = ""
        tool_payload: Any = None
        if delta is not None:
            text = delta.content or ""
            if delta.tool_calls:
                tool_payload = [t.model_dump(exclude_none=True) for t in delta.tool_calls]
        is_finished = finish is not None
        fr = finish or ("stop" if is_finished else "")
        chunk: GenericStreamingChunk = {
            "text": text,
            "is_finished": is_finished,
            "finish_reason": fr,
            "usage": usage if is_finished else None,
        }
        if tool_payload is not None:
            chunk["tool_use"] = tool_payload  # type: ignore[typeddict-item]
        out.append(chunk)
    return out


def register_chatgpt_litellm(
    chatgpt: ChatGPTClient | None = None,
    *,
    provider: str = CHATGPT_LITELLM_PROVIDER,
    replace: bool = True,
) -> ChatGPTLiteLLMAdapter:
    """Register ``provider/*`` models with LiteLLM so ``litellm.acompletion`` uses ``ChatGPTClient``.

    Call once at process startup before ``litellm.acompletion`` or ADK ``LiteLlm`` runs.
    Sets ``LITELLM_MODE=PRODUCTION`` by default so LiteLLM does not auto-load ``.env``.
    """
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
    handler = ChatGPTLiteLLMAdapter(chatgpt)
    if replace:
        litellm.custom_provider_map = [
            x for x in litellm.custom_provider_map if x.get("provider") != provider
        ]
        while provider in litellm._custom_providers:
            litellm._custom_providers.remove(provider)
        while provider in litellm.provider_list:
            litellm.provider_list.remove(provider)
    litellm.custom_provider_map.append({"provider": provider, "custom_handler": handler})
    custom_llm_setup()
    return handler


def build_adk_litellm_for_chatgpt(
    chatgpt: ChatGPTClient | None = None,
    *,
    openai_model_id: str | None = None,
):
    """Return ADK ``LiteLlm`` wired to ``ChatGPTClient`` (calls ``register_chatgpt_litellm``).

    Pass the result as ``LlmAgent(model=..., ...)`` — not as a bare string.

    ADK registers ``LiteLlm`` when ``litellm`` is installed
    (see ``google.adk.models``), but ``LLMRegistry`` only resolves *strings* that
    match ``LiteLlm.supported_models()`` (``openai/.*``, ``anthropic/.*``, …).
    The custom id ``sample_agent_chatgpt/<model>`` is **not** in that list, so
    ``LlmAgent(model="sample_agent_chatgpt/gpt-4.1-mini")`` would fail registry
    lookup. Instantiating ``LiteLlm(model=...)`` yourself bypasses that and
    still routes completions through LiteLLM → your ``custom_provider_map`` handler.
    """
    from google.adk.models.lite_llm import LiteLlm

    register_chatgpt_litellm(chatgpt)
    if openai_model_id and openai_model_id.strip():
        mid = f"{CHATGPT_LITELLM_PROVIDER}/{openai_model_id.strip()}"
    else:
        mid = default_litellm_model_id(chatgpt)
    return LiteLlm(model=mid)
