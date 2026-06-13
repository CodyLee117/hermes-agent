"""Tests for per-channel project-context injection (gateway/project_context.py)."""

import json
import pytest
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.project_context import (
    get_project_context_block,
    _reset_state_for_tests,
)


def _source(chat_id="100", thread_id=None):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_name="#test",
        chat_type="channel",
        user_id="42",
        user_name="cody",
        thread_id=thread_id,
    )


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a channel map + one project doc."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _reset_state_for_tests()
    (tmp_path / "project-rules").mkdir()
    (tmp_path / "channel-map.json").write_text(json.dumps({
        "channels": {"100": "galactic-cruise", "200": "sparrows"},
        "skip_channels": {"300": "multi-project"},
    }))
    (tmp_path / "project-rules" / "galactic-cruise.md").write_text(
        "# Galactic Cruise\n\nClient work. Use omnipotence."
    )
    (tmp_path / "project-rules" / "sparrows.md").write_text("# Sparrows\n\nUnity.")
    yield tmp_path
    _reset_state_for_tests()


class TestGetProjectContextBlock:
    def test_mapped_channel_injects_doc(self, hermes_home):
        block = get_project_context_block(_source("100"), "sess-1")
        assert "## Project Context (auto-loaded)" in block
        assert "### Project: galactic-cruise" in block
        assert "Use omnipotence." in block

    def test_unmapped_channel_is_noop(self, hermes_home):
        assert get_project_context_block(_source("999"), "sess-1") == ""

    def test_skip_channel_is_noop(self, hermes_home):
        # skip_channels entries are documentation only — not in "channels"
        assert get_project_context_block(_source("300"), "sess-1") == ""

    def test_additive_across_channels_same_session(self, hermes_home):
        get_project_context_block(_source("100"), "sess-1")
        block = get_project_context_block(_source("200"), "sess-1")
        # Both projects present once the session has touched both channels
        assert "### Project: galactic-cruise" in block
        assert "### Project: sparrows" in block

    def test_sessions_are_isolated(self, hermes_home):
        get_project_context_block(_source("100"), "sess-1")
        block = get_project_context_block(_source("999"), "sess-2")
        assert block == ""

    def test_idempotent_within_session(self, hermes_home):
        first = get_project_context_block(_source("100"), "sess-1")
        second = get_project_context_block(_source("100"), "sess-1")
        # Same content every turn — no duplicate sections accumulate
        assert first == second
        assert second.count("### Project: galactic-cruise") == 1

    def test_thread_id_also_matches(self, hermes_home):
        block = get_project_context_block(
            _source(chat_id="999", thread_id="100"), "sess-1"
        )
        assert "### Project: galactic-cruise" in block

    def test_missing_doc_is_noop(self, hermes_home):
        (hermes_home / "channel-map.json").write_text(json.dumps({
            "channels": {"100": "no-doc-project"},
        }))
        assert get_project_context_block(_source("100"), "sess-1") == ""

    def test_missing_map_is_noop(self, hermes_home, monkeypatch, tmp_path):
        (hermes_home / "channel-map.json").unlink()
        # Mask the shared ~/.hermes fallback so no real map leaks in
        monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
        assert get_project_context_block(_source("100"), "sess-1") == ""

    def test_malformed_map_is_noop(self, hermes_home):
        (hermes_home / "channel-map.json").write_text("{not json")
        assert get_project_context_block(_source("100"), "sess-1") == ""

    def test_unsafe_project_slug_rejected(self, hermes_home):
        (hermes_home / "channel-map.json").write_text(json.dumps({
            "channels": {"100": "../../etc/passwd"},
        }))
        assert get_project_context_block(_source("100"), "sess-1") == ""

    def test_oversize_doc_truncated(self, hermes_home):
        from gateway.project_context import _MAX_DOC_CHARS
        (hermes_home / "project-rules" / "galactic-cruise.md").write_text(
            "x" * (_MAX_DOC_CHARS + 1000)
        )
        block = get_project_context_block(_source("100"), "sess-1")
        assert len(block) < _MAX_DOC_CHARS + 500  # doc capped + small header

    def test_map_edit_picked_up_by_mtime_cache(self, hermes_home):
        import os, time
        assert get_project_context_block(_source("400"), "sess-1") == ""
        map_path = hermes_home / "channel-map.json"
        map_path.write_text(json.dumps({"channels": {"400": "sparrows"}}))
        # Ensure mtime actually changes on coarse-granularity filesystems
        os.utime(map_path, (time.time() + 2, time.time() + 2))
        block = get_project_context_block(_source("400"), "sess-1")
        assert "### Project: sparrows" in block

    def test_never_raises(self, hermes_home):
        # A source object missing attributes entirely must still fail open
        class Broken:
            pass
        assert get_project_context_block(Broken(), "sess-1") == ""


class TestProceduresLoader:
    """OMNI-079b: operating-procedures injection from the portal."""

    def _set_env(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_CHAT_URL", "http://portal")
        monkeypatch.setenv("WORKSPACE_CHAT_TOKEN", "kit-tok")
        monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")

    def _mock_urlopen(self, monkeypatch, procedures):
        import io
        from gateway import project_context as pc

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"procedures": procedures}).encode()

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            return _Resp()

        monkeypatch.setattr(pc.urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_no_env_no_block(self, hermes_home, monkeypatch):
        from gateway.project_context import _load_procedures_block
        # no WORKSPACE_CHAT_* env -> ""
        monkeypatch.delenv("WORKSPACE_CHAT_URL", raising=False)
        assert _load_procedures_block([]) == ""

    def test_loads_and_formats(self, hermes_home, monkeypatch):
        from gateway.project_context import _load_procedures_block
        self._set_env(monkeypatch)
        cap = self._mock_urlopen(monkeypatch, [
            {"title": "English only", "body": "Respond in English."},
            {"title": "No filler", "body": "No ack-only replies."},
        ])
        block = _load_procedures_block(["galactic-cruise"])
        assert "## Operating Procedures (auto-loaded)" in block
        assert "**English only**" in block and "Respond in English." in block
        assert "agent=kit" in cap["url"] and "type=hermes" in cap["url"]
        assert "project=galactic-cruise" in cap["url"]
        assert cap["auth"] == "Bearer kit-tok"

    def test_injected_even_without_project(self, hermes_home, monkeypatch):
        # an unmapped channel still loads global procedures
        self._set_env(monkeypatch)
        self._mock_urlopen(monkeypatch, [{"title": "Global rule", "body": "do X"}])
        block = get_project_context_block(_source("999"), "sess-proc")
        assert "## Operating Procedures (auto-loaded)" in block
        assert "**Global rule**" in block

    def test_fail_open_on_http_error(self, hermes_home, monkeypatch):
        from gateway import project_context as pc
        from gateway.project_context import _load_procedures_block
        self._set_env(monkeypatch)

        def boom(req, timeout=None):
            raise RuntimeError("portal down")
        monkeypatch.setattr(pc.urllib.request, "urlopen", boom)
        assert _load_procedures_block([]) == ""  # never raises
