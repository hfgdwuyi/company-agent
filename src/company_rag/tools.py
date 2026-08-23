# -*- coding: utf-8 -*-
"""Agent 工具注册表：企业知识库场景下的可调用工具集。

每个工具 = name + description + JSON Schema(parameters) + 执行函数。
"""
from __future__ import annotations

import ast
import datetime
import json
import logging
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import vector_store
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    examples: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolExecutionError(RuntimeError):
    pass


# ---------------- 工具实现 ----------------

def _safe_calc(expression: str) -> str:
    """安全数学计算：仅允许算术运算与基础函数。"""
    expr = (expression or "").strip()
    expr = re.sub(r"[xX×]", "*", expr)
    expr = re.sub(r"÷", "/", expr)
    allowed = {
        "add": operator.add, "sub": operator.sub, "mul": operator.mul,
        "div": operator.truediv, "pow": operator.pow, "mod": operator.mod,
        "abs": abs, "min": min, "max": max, "round": round,
    }
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise ToolExecutionError(f"不允许的标识符: {node.id}")
        elif not isinstance(
            node,
            (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
             ast.Call, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
             ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv, ast.Name),
        ):
            raise ToolExecutionError(f"不支持的表达式语法: {type(node).__name__}")
    def _eval(n: ast.AST):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return allowed[n.id]
        if isinstance(n, ast.BinOp):
            op = {
                ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                ast.FloorDiv: operator.floordiv,
            }[type(n.op)]
            return op(_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            op = {ast.USub: operator.neg, ast.UAdd: operator.pos}[type(n.op)]
            return op(_eval(n.operand))
        if isinstance(n, ast.Call):
            fn = allowed[n.func.id]
            args = [_eval(a) for a in n.args]
            return fn(*args)
        raise ToolExecutionError("不支持的表达式")
    try:
        result = _eval(tree.body)
        if isinstance(result, float):
            return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)
    except ZeroDivisionError:
        raise ToolExecutionError("除零错误") from None


def _retrieve_knowledge(question: str, top_k: int = 2) -> str:
    """检索知识库并返回相关片段（带来源）。结果按预算截断。"""
    from . import rag
    from .utils import truncate_to_budget

    hits = rag.retrieve(question, top_k=min(max(int(top_k or 2), 1), 5))
    if not hits:
        return "知识库中没有检索到相关资料。"
    parts = []
    for i, h in enumerate(hits, start=1):
        page = f"，第{h.page}页" if h.page else ""
        snippet = truncate_to_budget(h.text, 180)
        parts.append(
            f"[{i}] 《{h.filename}》{page} (相似度 {h.score:.3f})\n{snippet}"
        )
    text = "\n\n".join(parts)
    return truncate_to_budget(text, settings.llm_max_input_tokens - 120)


def _list_documents() -> str:
    docs = vector_store.list_documents(settings.default_collection)
    if not docs:
        return "知识库为空，还没有上传文档。"
    lines = ["当前知识库文档列表："]
    for d in docs:
        lines.append(f"- {d['filename']} (doc_id={d['doc_id']}, {d['chunk_count']} 个片段)")
    return "\n".join(lines)


def _get_document_summary(doc_id: str) -> str:
    """返回某文档的开头内容摘要（取前若干块）。"""
    chunks = vector_store.get_chunks_by_doc(settings.default_collection, doc_id, limit=5)
    if not chunks:
        return f"未找到文档 {doc_id}（或文档不在当前知识库中）。"
    head = "\n".join(c["text"][:300] for c in chunks[:3])
    return f"文档 {doc_id} 共 {len(chunks)}+ 个片段，开头内容：\n{head}"


def _get_current_datetime() -> str:
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %A')}（本地时区）"


def _count_documents() -> str:
    docs = vector_store.list_documents(settings.default_collection)
    return f"知识库中共有 {len(docs)} 份文档，{vector_store.count_chunks(settings.default_collection)} 个片段。"


# ---------------- 注册表 ----------------

def build_tools() -> list[Tool]:
    return [
        Tool(
            name="retrieve_knowledge",
            description="检索企业知识库中与问题相关的资料片段（向量检索）。问题涉及库内文档内容、数据、条款时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要检索的问题或关键词"},
                    "top_k": {"type": "integer", "description": "返回片段数量，默认 2"},
                },
                "required": ["question"],
            },
            func=_retrieve_knowledge,
            examples=["检索：公司员工考勤制度有哪些规定？", "检索：2024年研发投入"],
        ),
        Tool(
            name="calculator",
            description="数学计算（+ - * / 幂 括号 abs/min/max/round）。需要精确数值计算时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 (12.5*8+100)/4"},
                },
                "required": ["expression"],
            },
            func=_safe_calc,
            examples=["(12000*0.15+800)/12", "2**10"],
        ),
        Tool(
            name="get_current_datetime",
            description="获取当前日期时间（年/月/日/星期/时:分:秒）。问今天几号、星期几时使用。",
            parameters={"type": "object", "properties": {}},
            func=_get_current_datetime,
        ),
        Tool(
            name="list_documents",
            description='列出知识库全部文档（文件名、片段数）。问"有什么文档/资料"时使用。',
            parameters={"type": "object", "properties": {}},
            func=_list_documents,
        ),
        Tool(
            name="get_document_summary",
            description="查看某文档开头内容摘要（doc_id 用 list_documents 获取）。",
            parameters={
                "type": "object",
                "properties": {"doc_id": {"type": "string", "description": "文档 ID"}},
                "required": ["doc_id"],
            },
            func=_get_document_summary,
        ),
        Tool(
            name="count_documents",
            description="统计知识库文档数量与片段总数。",
            parameters={"type": "object", "properties": {}},
            func=_count_documents,
        ),
    ]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self.tools = {t.name: t for t in (tools if tools is not None else build_tools())}

    def names(self) -> list[str]:
        return list(self.tools.keys())

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tools.values()]

    def describe_for_prompt(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolExecutionError(f"未知工具: {name}，可用工具: {', '.join(self.names())}")
        try:
            result = tool.func(**arguments)
            if isinstance(result, (dict, list)):
                result = json.dumps(result, ensure_ascii=False)
            return str(result)
        except ToolExecutionError:
            raise
        except TypeError as e:
            raise ToolExecutionError(f"工具 {name} 参数错误: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise ToolExecutionError(f"工具 {name} 执行失败: {e}") from e


_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry
