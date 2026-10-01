"""Agent tools over the moment index, plus a small tool-calling loop.

Tools never raise into the agent: bad arguments come back as a structured error the
model can read and correct, the same way a real function-calling API would see them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .index import MomentIndex
from .synthetic import ACTIVITIES


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function definition."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _validate(params: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Minimal JSON-schema check: required keys, no unknown keys, basic types, enums."""
    props = params.get("properties", {})
    for key in params.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    types = {"integer": int, "number": (int, float), "string": str}
    for key, value in args.items():
        if key not in props:
            return f"unknown argument '{key}'"
        expected = types.get(props[key].get("type", ""))
        if expected and (not isinstance(value, expected) or isinstance(value, bool)):
            return f"argument '{key}' must be of type {props[key]['type']}"
        if "enum" in props[key] and value not in props[key]["enum"]:
            return f"argument '{key}' must be one of {props[key]['enum']}"
    return None


def build_tools(index: MomentIndex) -> dict[str, Tool]:
    session = {"type": "integer", "description": "Session id"}
    activity = {"type": "string", "enum": list(ACTIVITIES)}
    tools = [
        Tool(
            "find_moments",
            "Find time ranges where an activity happens, optionally in one session.",
            {
                "type": "object",
                "properties": {"activity": activity, "session_id": session, "min_confidence": {"type": "number"}},
                "required": ["activity"],
            },
            lambda activity, session_id=None, min_confidence=0.0: [
                m.as_dict() for m in index.find(activity, session_id, min_confidence)
            ],
        ),
        Tool(
            "session_timeline",
            "List everything that happened in a session, in order.",
            {"type": "object", "properties": {"session_id": session}, "required": ["session_id"]},
            lambda session_id: [m.as_dict() for m in index.timeline(session_id)],
        ),
        Tool(
            "activity_stats",
            "Seconds spent on each activity in a session.",
            {"type": "object", "properties": {"session_id": session}, "required": ["session_id"]},
            lambda session_id: index.stats(session_id),
        ),
        Tool(
            "similar_moments",
            "Find moments in other sessions that look and feel like a given moment.",
            {
                "type": "object",
                "properties": {"session_id": session, "time_s": {"type": "number"}, "k": {"type": "integer"}},
                "required": ["session_id", "time_s"],
            },
            lambda session_id, time_s, k=5: index.similar(session_id, float(time_s), k),
        ),
    ]
    return {t.name: t for t in tools}


def call_tool(tools: dict[str, Tool], name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool and always return ``{"ok": True, "result": ...}`` or ``{"ok": False, "error": ...}``."""
    tool = tools.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown tool '{name}'; available: {sorted(tools)}"}
    problem = _validate(tool.parameters, args)
    if problem:
        return {"ok": False, "error": problem}
    try:
        return {"ok": True, "result": tool.fn(**args)}
    except (KeyError, ValueError) as exc:
        return {"ok": False, "error": str(exc).strip("'\"")}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply: ...


@dataclass
class TraceStep:
    tool: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceStep]
    stopped_early: bool = False


SYSTEM_PROMPT = (
    "You answer questions about first-person recordings. Use the tools to look up moments; "
    "never guess times. Activities: " + ", ".join(ACTIVITIES) + ". If a tool returns an error, "
    "fix the arguments and try again. Answer briefly and include times in seconds."
)


class Agent:
    """Tool-calling loop: ask the model, run any tool calls, feed results back, repeat."""

    def __init__(self, llm: LLM, tools: dict[str, Tool], max_steps: int = 5) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        schemas = [t.schema() for t in self.tools.values()]
        trace: list[TraceStep] = []
        for _ in range(self.max_steps):
            reply = self.llm.complete(messages, schemas)
            if not reply.tool_calls:
                return AgentResult(reply.text or "", trace)
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                start = time.perf_counter()
                output = call_tool(self.tools, call.name, call.arguments)
                trace.append(TraceStep(call.name, call.arguments, output, (time.perf_counter() - start) * 1000))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})
        return AgentResult("I could not finish within the step limit.", trace, stopped_early=True)
