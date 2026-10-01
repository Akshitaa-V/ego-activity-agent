import json

import numpy as np
import pytest

from egoagent.agent import Agent, LLMReply, ToolCall, build_tools, call_tool
from egoagent.llm import OpenAICompatibleLLM, RuleBasedLLM, ScriptedLLM
from egoagent.vlm import ZeroShotLabeller


def test_timeline_merges_runs_without_overlap(toy_index):
    moments = toy_index.timeline(0)
    assert [(m.activity, m.start, m.end) for m in moments] == [("idle", 0, 3), ("stirring", 3, 5)]


def test_find_and_stats(toy_index):
    assert [m.session_id for m in toy_index.find("stirring")] == [0, 1]
    assert toy_index.stats(0) == {"idle": 3.0, "stirring": 2.0}
    with pytest.raises(ValueError):
        toy_index.find("dancing")


def test_similar_looks_only_in_other_sessions(toy_index):
    hits = toy_index.similar(0, 3.5, k=2)
    assert all(h["session_id"] == 1 for h in hits)
    assert hits[0]["similarity"] >= hits[1]["similarity"]


def test_tool_errors_are_returned_not_raised(toy_index):
    tools = build_tools(toy_index)
    assert call_tool(tools, "nope", {})["ok"] is False
    assert "missing required" in call_tool(tools, "session_timeline", {})["error"]
    assert "must be one of" in call_tool(tools, "find_moments", {"activity": "dancing"})["error"]
    assert "type integer" in call_tool(tools, "activity_stats", {"session_id": "0"})["error"]
    assert "unknown session" in call_tool(tools, "activity_stats", {"session_id": 99})["error"]
    assert call_tool(tools, "activity_stats", {"session_id": 0})["ok"] is True


def test_tool_schemas_are_openai_function_definitions(toy_index):
    for schema in (t.schema() for t in build_tools(toy_index).values()):
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


def test_agent_feeds_tool_errors_back_and_recovers(toy_index):
    llm = ScriptedLLM(
        [
            LLMReply(tool_calls=[ToolCall("1", "find_moments", {"activity": "cooking"})]),
            LLMReply(tool_calls=[ToolCall("2", "find_moments", {"activity": "stirring", "session_id": 0})]),
            LLMReply(text="You stirred from 3 to 5 s."),
        ]
    )
    result = Agent(llm, build_tools(toy_index)).run("when did I cook?")
    assert result.answer == "You stirred from 3 to 5 s."
    assert [s.output["ok"] for s in result.trace] == [False, True]
    tool_message = llm.seen[1][-1]
    assert tool_message["role"] == "tool" and json.loads(tool_message["content"])["ok"] is False


def test_agent_stops_at_step_limit(toy_index):
    calls = [LLMReply(tool_calls=[ToolCall(str(i), "activity_stats", {"session_id": 0})]) for i in range(10)]
    looping = ScriptedLLM(calls)
    result = Agent(looping, build_tools(toy_index), max_steps=3).run("loop")
    assert result.stopped_early and len(result.trace) == 3


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("When was I stirring in session 0?", "find_moments"),
        ("How long did I spend on each activity in session 0?", "activity_stats"),
        ("What happened in session 1?", "session_timeline"),
        ("Find moments similar to session 0 at 3.5s", "similar_moments"),
    ],
)
def test_rule_based_planner_routes_questions(toy_index, question, tool):
    result = Agent(RuleBasedLLM(), build_tools(toy_index)).run(question)
    assert [s.tool for s in result.trace] == [tool]
    assert result.trace[0].output["ok"] is True
    assert result.answer


def test_rule_based_planner_asks_for_missing_details(toy_index):
    result = Agent(RuleBasedLLM(), build_tools(toy_index)).run("hello?")
    assert result.trace == [] and "session" in result.answer


def test_openai_client_builds_request_and_parses_tool_calls():
    sent = {}

    def fake_transport(url, payload, headers):
        sent.update(url=url, payload=payload, headers=headers)
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "activity_stats", "arguments": '{"session_id": 2}'},
                            }
                        ],
                    }
                }
            ]
        }

    llm = OpenAICompatibleLLM(model="m", base_url="http://x/v1/", api_key="k", transport=fake_transport)
    reply = llm.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert sent["url"] == "http://x/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["payload"]["tools"] == [{"type": "function"}]
    assert reply.tool_calls[0].name == "activity_stats" and reply.tool_calls[0].arguments == {"session_id": 2}


def test_openai_client_survives_malformed_arguments():
    def bad(url, payload, headers):
        return {
            "choices": [
                {"message": {"tool_calls": [{"id": "c", "function": {"name": "find_moments", "arguments": "{oops"}}]}}
            ]
        }

    reply = OpenAICompatibleLLM(transport=bad).complete([], [])
    assert "_unparseable_arguments" in reply.tool_calls[0].arguments


class FakeClip:
    """Image i matches text i exactly; lets us test the scoring maths without weights."""

    def encode_texts(self, texts):
        return np.eye(len(texts), 8)

    def encode_images(self, frames):
        return np.tile(np.eye(5, 8)[2], (len(frames), 1))  # every frame looks like prompt 2


def test_zero_shot_labeller_averages_over_frames():
    labeller = ZeroShotLabeller(backend=FakeClip())
    scores = labeller.score_clip(np.zeros((4, 8, 8, 3), dtype=np.uint8))
    assert max(scores, key=scores.get) == "stirring"
    assert sum(scores.values()) == pytest.approx(1.0)
