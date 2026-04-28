"""Self-contained OpenAI chat client for SampleAgent."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

DEFAULT_MODEL = "gpt-4.1-mini"


class ChatGPTClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = (model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or None,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response.choices[0].message.content or ""

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletionMessage:
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, **kwargs
        )
        return response.choices[0].message
