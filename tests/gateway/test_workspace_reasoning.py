"""OMNI-068 slice 3: workspace reasoning routing.

- `_collect_reasoning` concatenates EVERY assistant step's reasoning (not just
  the last), so a multi-step agent loop's full thinking reaches the 💭 panel.
- WorkspaceAdapter stashes the turn's reasoning and POSTs it to the Cody-only
  panel after it sends the reply (it owns the reply's message_id). Best-effort:
  a POST failure never breaks delivery.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.run import _collect_reasoning
from gateway.config import Platform, PlatformConfig
from gateway.platforms.workspace import WorkspaceAdapter


def test_collect_reasoning_concatenates_all_steps():
    result = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "reasoning": "step one thinking", "content": ""},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "reasoning": "step two thinking", "content": "done"},
        ],
        "last_reasoning": "step two thinking",
    }
    out = _collect_reasoning(result)
    assert "step one thinking" in out and "step two thinking" in out
    assert out.index("step one") < out.index("step two")  # in order


def test_collect_reasoning_falls_back_to_last():
    # messages carry no per-message reasoning -> use last_reasoning
    result = {"messages": [{"role": "assistant", "content": "x"}],
              "last_reasoning": "only the last"}
    assert _collect_reasoning(result) == "only the last"


def test_collect_reasoning_empty():
    assert _collect_reasoning({"messages": [], "last_reasoning": None}) == ""
    assert _collect_reasoning({}) == ""


def _adapter():
    a = WorkspaceAdapter.__new__(WorkspaceAdapter)
    a._pending_reasoning = {}
    a.base_url = "http://portal"
    a.token = "kit-tok"
    return a


class _FakeResp:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeHttp:
    def __init__(self, status=200):
        self.status = status
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResp(self.status)


def test_flush_reasoning_posts_with_message_id():
    a = _adapter()
    a._http = _FakeHttp(200)
    a.stash_reasoning("3", "deep thoughts")
    asyncio.run(a._flush_reasoning("3", "229"))
    assert len(a._http.calls) == 1
    call = a._http.calls[0]
    assert call["url"] == "http://portal/api/chat/reasoning"
    assert call["json"] == {"message_id": 229, "text": "deep thoughts"}
    assert call["headers"]["Authorization"] == "Bearer kit-tok"
    assert a._pending_reasoning == {}  # consumed


def test_flush_reasoning_noop_without_stash():
    a = _adapter()
    a._http = _FakeHttp(200)
    asyncio.run(a._flush_reasoning("3", "229"))
    assert a._http.calls == []


def test_flush_reasoning_bad_message_id_skips():
    a = _adapter()
    a._http = _FakeHttp(200)
    a.stash_reasoning("3", "x")
    asyncio.run(a._flush_reasoning("3", ""))  # no id -> no POST, stash consumed
    assert a._http.calls == []
    assert a._pending_reasoning == {}


def test_flush_reasoning_swallows_errors():
    a = _adapter()

    class _Boom:
        def post(self, *args, **kwargs):
            raise RuntimeError("network down")

    a._http = _Boom()
    a.stash_reasoning("3", "x")
    # must not raise — delivery must never break on a reasoning POST
    asyncio.run(a._flush_reasoning("3", "229"))
