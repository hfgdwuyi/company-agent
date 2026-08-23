# -*- coding: utf-8 -*-
"""LLM 客户端：对接本地 Docker TensorRT-LLM（OpenAI 兼容 API）。

已探明的服务端行为（TRT-LLM 0.21 / trtllm-serve）：
1. 原生工具调用支持，但 tool_choice 必须是"具名函数"（不支持 auto/required）。
2. 工具参数以 <tool_call>…</tool_call> 包裹的 JSON 字符串返回。
3. 工具结果回传时，tool 角色消息不能带 name 字段（否则 400）。
4. model 字段必填但不校验值。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """解析 TRT-LLM 返回的工具参数：可能带 <tool_call> 包裹、或 {"name","arguments"} 双层结构。"""
    raw = (raw or "").strip()
    m = _TOOL_CALL_TAG.search(raw)
    if m:
        raw = m.group(1).strip()
    # 提取首个 { 到最后一个 } 之间的 JSON
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise LLMError(f"无法解析工具参数 JSON: {raw[:200]!r}") from None
    if not isinstance(data, dict):
        raise LLMError(f"工具参数不是 JSON 对象: {raw[:200]!r}")
    # TRT-LLM 双层结构 {"name": ..., "arguments": {...}} → 解包
    if set(data.keys()) == {"name", "arguments"} and isinstance(data.get("arguments"), dict):
        return data["arguments"]
    return data


def _extract_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    tcs = msg.get("tool_calls") or []
    out: list[ToolCall] = []
    for tc in tcs:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "") or ""
        try:
            args = parse_tool_arguments(raw_args)
        except LLMError:
            args = {}
        out.append(ToolCall(id=tc.get("id", ""), name=name, arguments=args, raw_arguments=raw_args))
    return out


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_concurrency: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.timeout = timeout or settings.llm_timeout
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_concurrency = max_concurrency or settings.llm_max_concurrency
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout)
        self._sem = asyncio.Semaphore(self.max_concurrency)

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        r = await self._client.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        extra: Optional[dict] = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if extra:
            payload.update(extra)

        async with self._sem:
            if stream:
                return await self._chat_stream(payload)
            return await self._chat_once(payload)

    async def _chat_once(self, payload: dict) -> ChatResult:
        for attempt in range(2):
            try:
                r = await self._client.post("/chat/completions", json=payload)
                if r.status_code >= 500 and attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                return ChatResult(
                    content=msg.get("content") or "",
                    tool_calls=_extract_tool_calls(msg),
                    finish_reason=choice.get("finish_reason", ""),
                    usage=data.get("usage", {}),
                    raw=data,
                )
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    detail = e.response.json().get("message", "")
                except Exception:  # noqa: BLE001
                    detail = e.response.text[:300]
                if e.response.status_code == 400:
                    with open("last_failed_payload.json", "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                raise LLMError(f"LLM HTTP {e.response.status_code}: {detail}") from e
            except httpx.TimeoutException as e:
                if attempt == 0:
                    logger.warning("LLM 超时，重试一次: %s", e)
                    continue
                raise LLMError(f"LLM 请求超时({self.timeout}s)") from e
        raise LLMError("LLM 请求失败")  # pragma: no cover

    async def _chat_stream(self, payload: dict) -> ChatResult:
        """流式：聚合 SSE 增量，返回完整结果（含可能的 tool_calls）。"""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = ""
        async with self._client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = evt.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content")
                if c:
                    content_parts.append(c)
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        fn = tc.get("function") or {}
                        # 流式分段拼接
                        idx = tc.get("index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append(ToolCall(id="", name="", arguments={}, raw_arguments=""))
                        cur = tool_calls[idx]
                        if tc.get("id"):
                            cur.id = tc["id"]
                        if fn.get("name"):
                            cur.name = fn["name"]
                        cur.raw_arguments += fn.get("arguments") or ""
                        if tc.get("finish") is not None:  # pragma: no cover - 兼容性
                            pass
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
        for tc in tool_calls:
            if tc.raw_arguments:
                try:
                    tc.arguments = parse_tool_arguments(tc.raw_arguments)
                except LLMError:
                    pass
        return ChatResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def stream_events(self, payload: dict) -> AsyncIterator[dict]:
        """真正的 SSE 事件流：逐条产出 {"delta": str} / {"tool_calls": [...]} / {"done": True}。

        供 API 层做 StreamingResponse 透传（token 级流式）。
        """
        tool_calls: list[dict] = []
        async with self._sem:
            async with self._client.stream("POST", "/chat/completions", json=payload) as r:
                if r.status_code >= 400:
                    detail = (await r.aread()).decode("utf-8", errors="replace")[:500]
                    raise LLMError(f"LLM HTTP {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        evt = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = evt.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    c = delta.get("content")
                    if c:
                        yield {"type": "delta", "delta": c}
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({"id": "", "name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            if fn.get("name"):
                                tool_calls[idx]["name"] = fn["name"]
                            tool_calls[idx]["arguments"] += fn.get("arguments") or ""
                    if choices[0].get("finish_reason"):
                        yield {"type": "finish", "finish_reason": choices[0]["finish_reason"]}
        if tool_calls:
            parsed = []
            for tc in tool_calls:
                try:
                    args = parse_tool_arguments(tc["arguments"])
                except LLMError:
                    args = {}
                parsed.append(
                    {"id": tc["id"], "name": tc["name"], "arguments": args, "raw_arguments": tc["arguments"]}
                )
            yield {"type": "tool_calls", "tool_calls": parsed}
        yield {"type": "done"}


_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
