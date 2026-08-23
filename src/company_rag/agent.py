# -*- coding: utf-8 -*-
"""Agent：基于 TRT-LLM 原生工具调用的智能体。

服务端限制：tool_choice 仅支持"具名函数"。因此采用两阶段协议：
  1) 决策轮：模型从工具清单里选一个工具（JSON 输出），或直接回答；
  2) 强制轮：用 tool_choice=<具名工具> 走服务端原生工具调用，获得结构化参数；
  3) 执行工具 → 以 tool 角色消息回传结果 → 回到决策轮（最多 N 轮）；
  4) 无工具需求时输出最终答案。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Optional

from .config import settings
from .llm_client import LLMClient, LLMError, ToolCall, get_llm_client
from .models import ChatResponse, ChatMessage, ToolInvocation
from .tools import ToolRegistry, ToolExecutionError, get_registry
from .utils import estimate_tokens, truncate_to_budget

logger = logging.getLogger(__name__)

DECISION_PROMPT = (
    "你是智能助手。可用工具：\n{tools}\n\n"
    "需要工具时只输出JSON：{{\"tool\":\"工具名\",\"arguments\":{{...}}}}；"
    "不需要工具则直接回答（中文）。拿不准就调用 retrieve_knowledge。\n"
    "若问题含多个独立子问题，可多次调用工具（每次输出一个）。\n"
    "若工具结果已足够回答，请直接回答，不要重复调用同一工具。\n"
)

_JSON_RE = re.compile(r"\{.*\}", re.S)

# 轻量意图识别（确定性兜底：复合问题提示候选工具）
_INTENT_RULES: list[tuple[str, str]] = [
    (r"几份文档|多少份文档|文档数量|文档列表|有哪些文档|有什么文档|统计.*文档|count", "count_documents"),
    (r"文档|知识库|资料|论文|规定|要求|条款|制度|内容", "retrieve_knowledge"),
    (r"[0-9０-９].*[+\-*/×xX÷]|[+\-*/×xX÷].*[0-9０-９]|算一下|计算|等于多少|多少倍", "calculator"),
    (r"几号|星期|今天.*时间|现在.*时间|几点了|日期", "get_current_datetime"),
]


def detect_intents(message: str) -> list[str]:
    hits: list[str] = []
    for pat, tool in _INTENT_RULES:
        if re.search(pat, message) and tool not in hits:
            hits.append(tool)
    return hits


def _trim_history(messages: list[dict], budget_tokens: int) -> None:
    """裁剪对话历史（保留首条用户问题与最新消息），保证决策轮输入在预算内。"""
    if len(messages) <= 2:
        return
    while len(messages) > 2 and estimate_tokens(
        json.dumps(messages, ensure_ascii=False)
    ) > budget_tokens:
        # 成对删除中间历史（assistant tool_calls + tool 结果），保留首条问题与最新消息
        if len(messages) >= 3 and messages[1].get("role") == "assistant":
            del messages[1:3]
        else:
            messages.pop(1)


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中容错提取 JSON 对象。"""
    if not text:
        return None
    # 去掉 markdown 代码围栏
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # 容错：尝试逐行找 tool 字段
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('"tool"') or line.startswith("tool"):
                try:
                    return json.loads("{" + line + "}")
                except json.JSONDecodeError:
                    continue
        return None


class Agent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        max_iterations: Optional[int] = None,
    ):
        self.llm = llm or get_llm_client()
        self.registry = registry or get_registry()
        self.max_iterations = max_iterations or settings.agent_max_iterations

    # ---------- 决策轮 ----------
    async def _decide(self, messages: list[dict], hint_tools: Optional[list[str]] = None) -> tuple[Optional[str], Optional[dict]]:
        """返回 (tool_name | None, arguments | None)。None 表示直接回答。"""
        tool_desc = self.registry.describe_for_prompt()
        system = DECISION_PROMPT.format(tools=tool_desc)
        if hint_tools:
            system += f"\n提示：此问题可能包含多个子任务，请依次调用可用工具：{', '.join(hint_tools)}。"
        decision_msgs = [
            {"role": "system", "content": system},
            *messages,
        ]
        result = await self.llm.chat(decision_msgs, max_tokens=settings.agent_max_tool_tokens, temperature=0.0)
        text = result.content.strip()
        if not text:
            return None, None
        data = _extract_json(text)
        if data and data.get("tool"):
            args = data.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            return str(data["tool"]), args
        return None, None

    # ---------- 强制轮（服务端原生工具调用） ----------
    async def _force_tool_call(self, messages: list[dict], tool_name: str) -> Optional[ToolCall]:
        # 只传目标工具的 schema（tools 定义会占用输入 token 预算，全量 6 个会超限）
        tool = self.registry.get(tool_name)
        schemas = [tool.to_openai_schema()] if tool else []
        tool_choice = {"type": "function", "function": {"name": tool_name}}
        result = await self.llm.chat(
            messages,
            tools=schemas,
            tool_choice=tool_choice,
            max_tokens=settings.agent_max_tool_tokens,
            temperature=0.0,
        )
        if not result.tool_calls:
            # 模型拒绝调用：把选择轮参数当作用户意图，尝试直接执行
            logger.warning("模型未产出工具调用（强制 %s），降级处理", tool_name)
            return None
        return result.tool_calls[0]

    # ---------- 预算控制 ----------
    def _fit_budget(self, messages: list[dict]) -> None:
        """让 messages（不含 system）在决策轮总预算内：先截断最近工具结果，再成对裁剪历史。"""
        sys_tokens = estimate_tokens(DECISION_PROMPT.format(tools=self.registry.describe_for_prompt()))
        budget = settings.llm_max_input_tokens - sys_tokens - 20
        # 截断最近的 tool 结果（保留可读性下限 60 tokens）
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("content"):
                m["content"] = truncate_to_budget(m["content"], max(60, budget - 90))
                break
        _trim_history(messages, budget)

    # ---------- 主流程 ----------
    async def run(
        self,
        message: str,
        history: Optional[list[ChatMessage]] = None,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        session_id = session_id or uuid.uuid4().hex[:12]
        messages: list[dict] = [{"role": m.role, "content": m.content} for m in (history or [])]
        messages.append({"role": "user", "content": message})
        hint_tools = detect_intents(message)
        hint_tools = hint_tools if len(hint_tools) > 1 else None  # 仅复合问题注入提示

        invocations: list[ToolInvocation] = []
        final_content: Optional[str] = None

        for _ in range(self.max_iterations):
            self._fit_budget(messages)
            tool_name, _ = await self._decide(messages, hint_tools=hint_tools)
            if tool_name is None:
                break
            if self.registry.get(tool_name) is None:
                # 未知工具：告知模型并继续
                messages.append({"role": "user", "content": f"工具 {tool_name} 不存在，可用工具：{', '.join(self.registry.names())}。请重新选择。"})
                continue

            # 重复调用守卫：同一工具已执行过则不再执行，直接进入最终回答
            if any(inv.name == tool_name for inv in invocations):
                logger.info("Agent 重复选择工具 %s，跳过执行，直接回答", tool_name)
                break
            tc = await self._force_tool_call(messages, tool_name)
            if tc is None:
                final_content = f"（工具调用失败：模型未能生成 {tool_name} 的参数）"
                break
            # 执行
            started = time.perf_counter()
            ok, result_text = True, ""
            try:
                result_text = self.registry.execute(tc.name or tool_name, tc.arguments)
            except ToolExecutionError as e:
                ok, result_text = False, str(e)
            elapsed = int((time.perf_counter() - started) * 1000)
            logger.info("Agent 工具调用 %s(%s) -> %s (%dms)", tc.name, tc.arguments, "OK" if ok else "FAIL", elapsed)

            # 回传历史：assistant 的 tool_calls + tool 结果（注意：tool 消息不带 name，服务端限制）
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc.id or f"call_{len(invocations)}",
                        "type": "function",
                        "function": {
                            "name": tc.name or tool_name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
            messages.append(assistant_msg)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id or f"call_{len(invocations)}", "content": result_text}
            )
            # 预算约束：截断工具结果 + 裁剪历史（供下一轮决策）
            self._fit_budget(messages)
            fitted_result = result_text
            for m in reversed(messages):
                if m.get("role") == "tool":
                    fitted_result = m.get("content", "")
                    break
            invocations.append(
                ToolInvocation(name=tc.name or tool_name, arguments=tc.arguments, result=fitted_result, ok=ok, elapsed_ms=elapsed)
            )

        # 最终回答：把工具结果显式内联进用户消息（7B 模型对 role=tool 消息的遵循能力弱）
        if final_content is None:
            if invocations:
                results_text = "\n".join(
                    f"【{inv.name}】{inv.result[:350]}" for inv in invocations
                )
                results_text = truncate_to_budget(
                    results_text, settings.llm_max_input_tokens - 160
                )
                final_msgs = [
                    {"role": "system", "content": "你是企业知识库助手。请基于工具返回结果回答用户问题，直接给出答案；工具结果中没有的信息不要编造。回答控制在 150 字以内，简洁扼要。"},
                    {"role": "user", "content": f"问题：{message}\n\n工具返回结果：\n{results_text}"},
                ]
            else:
                final_msgs = [
                    {"role": "system", "content": "你是企业知识库助手。请直接回答用户问题。"},
                    {"role": "user", "content": message},
                ]
            result = await self.llm.chat(final_msgs, temperature=0.3)
            final_content = result.content.strip()

        return ChatResponse(
            session_id=session_id,
            answer=final_content,
            tool_invocations=invocations,
            usage={},
        )


    async def run_stream(
        self,
        message: str,
        history: Optional[list[ChatMessage]] = None,
        session_id: Optional[str] = None,
    ):
        """流式版：先跑工具循环（产出 tool 事件），最终答案走 token 级 SSE。"""
        session_id = session_id or uuid.uuid4().hex[:12]
        messages: list[dict] = [{"role": m.role, "content": m.content} for m in (history or [])]
        messages.append({"role": "user", "content": message})
        _stream_invocations: list[dict] = []
        hint_tools = detect_intents(message)
        hint_tools = hint_tools if len(hint_tools) > 1 else None  # 仅复合问题注入提示

        for _ in range(self.max_iterations):
            self._fit_budget(messages)
            tool_name, _ = await self._decide(messages, hint_tools=hint_tools)
            if tool_name is None:
                break
            if self.registry.get(tool_name) is None:
                messages.append({"role": "user", "content": f"工具 {tool_name} 不存在，可用工具：{', '.join(self.registry.names())}。请重新选择。"})
                continue
            if any(inv.get("tool") == tool_name for inv in _stream_invocations):
                logger.info("Agent 流式重复选择工具 %s，跳过执行，直接回答", tool_name)
                break

            tc = await self._force_tool_call(messages, tool_name)
            if tc is None:
                yield {"type": "tool_error", "tool": tool_name, "error": "模型未能生成工具参数"}
                break
            started = time.perf_counter()
            ok, result_text = True, ""
            try:
                result_text = self.registry.execute(tc.name or tool_name, tc.arguments)
            except ToolExecutionError as e:
                ok, result_text = False, str(e)
            elapsed = int((time.perf_counter() - started) * 1000)
            _stream_invocations.append({"tool": tc.name or tool_name, "result": result_text})
            yield {
                "type": "tool",
                "tool": tc.name or tool_name,
                "arguments": tc.arguments,
                "result": result_text,
                "ok": ok,
                "elapsed_ms": elapsed,
            }
            call_id = tc.id or f"call_{elapsed}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name or tool_name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result_text})
            self._fit_budget(messages)

        # 最终回答（流式）：工具结果显式内联
        if _stream_invocations:
            results_text = "\n".join(
                f"【{inv['tool']}】{str(inv.get('result', ''))[:350]}" for inv in _stream_invocations
            )
            results_text = truncate_to_budget(results_text, settings.llm_max_input_tokens - 160)
            final_messages = [
                {"role": "system", "content": "你是企业知识库助手。请基于工具返回结果回答用户问题，直接给出答案；工具结果中没有的信息不要编造。回答控制在 150 字以内，简洁扼要。"},
                {"role": "user", "content": f"问题：{message}\n\n工具返回结果：\n{results_text}"},
            ]
        else:
            final_messages = [
                {"role": "system", "content": "你是企业知识库助手。请直接回答用户问题。"},
                {"role": "user", "content": message},
            ]
        payload = {
            "model": self.llm.model,
            "messages": final_messages,
            "max_tokens": self.llm.max_tokens,
            "temperature": 0.3,
            "stream": True,
        }
        async for evt in self.llm.stream_events(payload):
            if evt.get("type") == "done":
                continue  # 由下面统一发 done（带 session_id）
            yield evt
        yield {"type": "done", "session_id": session_id}


_default_agent: Optional[Agent] = None


def get_agent() -> Agent:
    global _default_agent
    if _default_agent is None:
        _default_agent = Agent()
    return _default_agent
