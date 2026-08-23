# -*- coding: utf-8 -*-
"""Probe TRT-LLM OpenAI-compatible endpoint capabilities (tool calling round-trip)."""
import json
import sys

import httpx

BASE = "http://127.0.0.1:8001/v1"
MODEL = "engine_qwen_int4_v2"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市当前的天气情况",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                "required": ["city"],
            },
        },
    }
]


def chat(messages, tools=None, tool_choice=None, max_tokens=512, temperature=0.0):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    r = httpx.post(f"{BASE}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def main():
    # 1. plain chat
    print("=== 1. plain chat ===")
    msg = chat([{"role": "user", "content": "用一句话介绍你自己"}])
    print(json.dumps(msg, ensure_ascii=False, indent=2))

    # 2. forced named tool call
    print("\n=== 2. forced named tool call ===")
    msg2 = chat(
        [{"role": "user", "content": "查询北京的天气，使用工具。"}],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    print(json.dumps(msg2, ensure_ascii=False, indent=2))

    # 3. round trip: feed tool result back, ask final answer
    print("\n=== 3. tool result round-trip ===")
    tc = msg2.get("tool_calls") or []
    if tc:
        tool_call = tc[0]
        history = [
            {"role": "user", "content": "查询北京的天气，使用工具。"},
            {"role": "assistant", "content": msg2.get("content") or "", "tool_calls": tc},
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "content": "北京今天晴，气温 25 度，东南风 3 级",
            },
        ]
        msg3 = chat(history, tools=TOOLS)
        print(json.dumps(msg3, ensure_ascii=False, indent=2))
    else:
        print("no tool call produced in step 2, skipping round trip")


if __name__ == "__main__":
    sys.exit(main())
