"""OMNI-200 S2b (DDR-0036 P-6) — cap-out work-preservation + dispatch autopsy.

The harness fail-safe that runs at the agent loop's iteration-cap: find the agent's own
working dispatches (OMNI-198 scoped lookup), HEAD-safely preserve uncommitted work to a
pushed recovery ref, then file the OMNI-200 S1 autopsy. Best-effort; never raises."""
import subprocess
from pathlib import Path

import pytest

import tools.cap_autopsy as ca


# ── file_cap_autopsy: orchestration (HTTP + preserve mocked) ──────────────────

def test_noop_without_workspace_config():
    assert ca.file_cap_autopsy("", "", "", "/tmp") == []
    assert ca.file_cap_autopsy("kit", "", "tok", "/tmp") == []  # missing base_url


def test_autopsies_each_working_dispatch(monkeypatch):
    calls = []

    def fake_api(method, url, token, body=None):
        calls.append((method, url, body))
        if method == "GET":
            return {"dispatches": [{"id": 11}, {"id": 12}]}
        return {"dispatch": {"id": 99, "state": "dead"}}

    monkeypatch.setattr(ca, "_api", fake_api)
    monkeypatch.setattr(ca, "_preserve_work", lambda cwd, branch: branch)
    out = ca.file_cap_autopsy("kit", "https://p", "tok", "/repo")
    assert out == [11, 12]
    posts = [c for c in calls if c[0] == "POST"]
    assert len(posts) == 2
    assert posts[0][2]["reason"] == "iteration_cap"
    assert posts[0][2]["recovery_ref"]  # the preserved branch flowed through
    assert "/dispatches/11/autopsy" in posts[0][1]


def test_lookup_uses_agent_scoped_working_filter(monkeypatch):
    seen = {}
    monkeypatch.setattr(ca, "_api",
                        lambda m, u, t, body=None: seen.update(url=u) or {"dispatches": []})
    ca.file_cap_autopsy("kit", "https://p", "tok", "/repo")
    assert "state=working" in seen["url"]


def test_autopsy_tolerates_409(monkeypatch):
    import urllib.error

    def fake_api(method, url, token, body=None):
        if method == "GET":
            return {"dispatches": [{"id": 7}]}
        raise urllib.error.HTTPError(url, 409, "already done", {}, None)

    monkeypatch.setattr(ca, "_api", fake_api)
    monkeypatch.setattr(ca, "_preserve_work", lambda cwd, branch: None)
    assert ca.file_cap_autopsy("kit", "https://p", "tok", "/repo") == []  # 409 → not counted, no raise


def test_lookup_failure_is_swallowed(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(ca, "_api", boom)
    assert ca.file_cap_autopsy("kit", "https://p", "tok", "/repo") == []  # never raises


def test_preserve_returns_none_keeps_recovery_ref_none(monkeypatch):
    posted = {}
    def fake_api(method, url, token, body=None):
        if method == "GET":
            return {"dispatches": [{"id": 5}]}
        posted.update(body or {})
        return {}
    monkeypatch.setattr(ca, "_api", fake_api)
    monkeypatch.setattr(ca, "_preserve_work", lambda cwd, branch: None)
    ca.file_cap_autopsy("kit", "https://p", "tok", "/repo")
    assert posted["recovery_ref"] is None
    assert "untracked files not captured" in posted["note"]


# ── _preserve_work: real git, HEAD-safe (the shared-clone constraint) ─────────

def _git(cwd, *a):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)


@pytest.fixture()
def repo_with_remote(tmp_path):
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True)
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    _git(work, "config", "commit.gpgsign", "false")
    (work / "f.txt").write_text("v1\n")
    _git(work, "add", "f.txt")
    _git(work, "commit", "-m", "init")
    _git(work, "remote", "add", "origin", str(bare))
    return work, bare


def test_preserve_clean_tree_returns_none(repo_with_remote):
    work, _ = repo_with_remote
    assert ca._preserve_work(str(work), "recovery/x") is None


def test_preserve_pushes_recovery_ref_without_switching_head(repo_with_remote):
    work, bare = repo_with_remote
    head_before = _git(work, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    (work / "f.txt").write_text("v2-uncommitted\n")  # dirty tracked change
    out = ca._preserve_work(str(work), "recovery/kit-d1")
    assert out == "recovery/kit-d1"
    # the recovery ref exists on the remote
    refs = _git(bare, "branch", "--list", "recovery/kit-d1").stdout
    assert "recovery/kit-d1" in refs
    # HEAD-safe: still on the same branch, working tree still holds the change
    assert _git(work, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == head_before
    assert (work / "f.txt").read_text() == "v2-uncommitted\n"


def test_preserve_non_git_dir_returns_none(tmp_path):
    assert ca._preserve_work(str(tmp_path), "recovery/x") is None


# ── root gate on the agent hook (subagent must NOT autopsy parent's dispatch) ──

from types import SimpleNamespace  # noqa: E402

from run_agent import AIAgent  # noqa: E402


def _set_ws_env(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_URL", "https://p")
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    monkeypatch.setenv("WORKSPACE_CHAT_TOKEN", "tok")


def test_hook_fires_for_root(monkeypatch):
    called = []
    monkeypatch.setattr(ca, "file_cap_autopsy", lambda *a, **k: called.append(a) or [])
    _set_ws_env(monkeypatch)
    AIAgent._file_cap_autopsy_if_root(SimpleNamespace(parent_session_id=None))
    assert called and called[0][0] == "kit"


def test_hook_skips_subagent(monkeypatch):
    called = []
    monkeypatch.setattr(ca, "file_cap_autopsy", lambda *a, **k: called.append(a) or [])
    _set_ws_env(monkeypatch)
    AIAgent._file_cap_autopsy_if_root(SimpleNamespace(parent_session_id="parent-abc"))
    assert called == []


def test_hook_skips_without_workspace_env(monkeypatch):
    called = []
    monkeypatch.setattr(ca, "file_cap_autopsy", lambda *a, **k: called.append(a) or [])
    for k in ("WORKSPACE_CHAT_URL", "WORKSPACE_CHAT_AGENT", "WORKSPACE_CHAT_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    AIAgent._file_cap_autopsy_if_root(SimpleNamespace(parent_session_id=None))
    assert called == []


def test_hook_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(ca, "file_cap_autopsy", boom)
    _set_ws_env(monkeypatch)
    # must swallow — a fail-safe cannot crash the agent's shutdown
    AIAgent._file_cap_autopsy_if_root(SimpleNamespace(parent_session_id=None))
