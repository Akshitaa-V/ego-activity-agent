"""Language-model backends for the agent.

- ``RuleBasedLLM``: a deterministic keyword planner so the demo runs offline with no API key.
  It is a stand-in, not a language model, and is labelled that way in the README.
- ``ScriptedLLM``: replays fixed replies; used in tests.
- ``OpenAICompatibleLLM``: any ``/chat/completions`` endpoint with tool calling (OpenAI,
  vLLM, Ollama, a university cluster). The HTTP transport is injectable for testing.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from .agent import LLMReply, ToolCall
from .synthetic import ACTIVITIES

ACTIVITY_WORDS = {
    "idle": "idle",
    "resting": "idle",
    "still": "idle",
    "walking": "walking",
    "walk": "walking",
    "stirring": "stirring",
    "stir": "stirring",
    "cooking": "stirring",
    "typing": "typing",
    "type": "typing",
    "keyboard": "typing",
    "reaching": "reaching",
    "reach": "reaching",
    "grab": "reaching",
}


class RuleBasedLLM:
    """Keyword planner: picks one tool from the question, then summarises its output."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        last = messages[-1]
        if last["role"] == "tool":
            return LLMReply(text=self._summarise(json.loads(last["content"]), messages))
        return self._plan(last["content"])

    def _plan(self, question: str) -> LLMReply:
        q = question.lower()
        session = re.search(r"session\s*(\d+)", q)
        time_s = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec|econds)?\b", q)
        activity = next((ACTIVITY_WORDS[w] for w in re.findall(r"[a-z]+", q) if w in ACTIVITY_WORDS), None)
        args: dict[str, Any] = {}
        if session:
            args["session_id"] = int(session.group(1))
        if ("similar" in q or "like" in q) and session and time_s:
            name = "similar_moments"
            args["time_s"] = float(time_s.group(1))
        elif ("how long" in q or "how much time" in q) and session:
            name = "activity_stats"
        elif activity:
            name = "find_moments"
            args["activity"] = activity
        elif session:
            name = "session_timeline"
        else:
            hint = "Please name a session (e.g. 'session 3') or an activity: " + ", ".join(ACTIVITIES) + "."
            return LLMReply(text=hint)
        return LLMReply(tool_calls=[ToolCall("call_1", name, args)])

    @staticmethod
    def _summarise(output: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        if not output.get("ok"):
            return f"The lookup failed: {output.get('error')}"
        result = output["result"]
        if isinstance(result, dict):
            return "Time per activity: " + ", ".join(f"{k} {v}s" for k, v in result.items()) + "."
        if not result:
            return "No matching moments found."
        first = result[0]
        if "similarity" in first:
            parts = [
                f"session {r['session_id']} at {r['start_s']}s ({r['activity']}, sim {r['similarity']})" for r in result
            ]
            return "Most similar moments: " + "; ".join(parts) + "."
        parts = [f"{m['start_s']}-{m['end_s']}s {m['activity']} (session {m['session_id']})" for m in result[:8]]
        more = f" and {len(result) - 8} more" if len(result) > 8 else ""
        return f"Found {len(result)} moment(s): " + "; ".join(parts) + more + "."


class ScriptedLLM:
    """Returns the given replies in order and records what it was sent."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else LLMReply(text="(no more scripted replies)")


Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def _http_post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"refusing non-HTTP URL: {url}")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - scheme checked above
        return json.loads(resp.read())


class OpenAICompatibleLLM:
    """Chat-completions client with tool calling. Configure with environment variables
    ``EGOAGENT_BASE_URL``, ``EGOAGENT_MODEL`` and ``EGOAGENT_API_KEY`` or pass them in."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport = _http_post,
    ) -> None:
        self.model = model or os.environ.get("EGOAGENT_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("EGOAGENT_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("EGOAGENT_API_KEY", "")
        self.transport = transport

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0}
        data = self.transport(f"{self.base_url}/chat/completions", payload, headers)
        message = data["choices"][0]["message"]
        calls = []
        for c in message.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": c["function"].get("arguments")}
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        return LLMReply(text=message.get("content"), tool_calls=calls)
