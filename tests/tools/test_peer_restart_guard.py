"""OMNI-200 S2 (DDR-0036 P-6) — crew peer-restart guard.

Crew agents run as `hermes-<profile>.service` user units under ONE OS user, so any
agent can restart/kill a peer (the 2026-06-29 cascade). The crew run
`approvals.mode: off`, which bypasses every other guard, so this guard is a HARD,
UNBYPASSABLE block that runs before the yolo/off bypass — but only in gateway
(autonomous) sessions; interactive CLI keeps the normal approval flow."""
import os

import pytest

from tools.approval import check_all_command_guards, check_peer_restart_guard


@pytest.fixture()
def as_kit(monkeypatch):
    """Run as crew agent 'kit' in a gateway session."""
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)


# ── unit: check_peer_restart_guard (identity via WORKSPACE_CHAT_AGENT) ────────

def test_blocks_systemctl_restart_of_peer(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("systemctl --user restart hermes-forge") is not None


def test_blocks_systemctl_stop_of_dispatcher(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("systemctl --user stop ziggy-gateway") is not None


def test_blocks_systemctl_kill_of_chat_leg(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "forge")
    assert check_peer_restart_guard("systemctl --user kill ziggy-chat-leg") is not None


def test_allows_restart_of_own_service(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("systemctl --user restart hermes-kit") is None


def test_allows_readonly_status_of_peer(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("systemctl --user status hermes-forge") is None
    assert check_peer_restart_guard("journalctl --user -u hermes-forge -n 50") is None


def test_blocks_pkill_hermes_broadly(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("pkill -9 -f hermes") is not None
    assert check_peer_restart_guard("killall gateway") is not None


def test_blocks_pkill_of_named_peer(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("pkill -f forge") is not None


def test_fails_closed_when_identity_unknown(monkeypatch):
    """Indeterminate identity → every agent service is treated as a peer."""
    monkeypatch.delenv("WORKSPACE_CHAT_AGENT", raising=False)
    monkeypatch.setenv("HERMES_HOME", "/nonexistent/profiles/none")
    assert check_peer_restart_guard("systemctl --user restart hermes-kit") is not None


def test_ignores_unrelated_commands(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    assert check_peer_restart_guard("git status") is None
    assert check_peer_restart_guard("systemctl --user restart my-app.service") is None


# ── integration: hard block survives approvals.mode=off (the real failure) ────

def test_hard_block_in_gateway_survives_off_mode(as_kit, monkeypatch):
    """The 2026-06-29 reproduction: even with approvals.mode=off (which returns
    approved for every other guard), the peer-restart is refused."""
    monkeypatch.setattr("tools.approval._get_approval_mode", lambda: "off")
    res = check_all_command_guards("systemctl --user restart hermes-forge", "local")
    assert res["approved"] is False
    assert res.get("peer_restart_blocked") is True


def test_own_service_restart_passes_in_gateway(as_kit, monkeypatch):
    monkeypatch.setattr("tools.approval._get_approval_mode", lambda: "off")
    res = check_all_command_guards("systemctl --user restart hermes-kit", "local")
    assert res["approved"] is True


def test_containers_bypass_peer_guard(as_kit):
    # a container can't reach host systemd units → guard is moot
    res = check_all_command_guards("systemctl --user restart hermes-forge", "docker")
    assert res["approved"] is True


def test_non_gateway_not_hard_blocked(monkeypatch):
    """Interactive CLI (no gateway session) is not hard-blocked here — it keeps the
    normal approval flow (off-mode returns approved)."""
    monkeypatch.setenv("WORKSPACE_CHAT_AGENT", "kit")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr("tools.approval._get_approval_mode", lambda: "off")
    res = check_all_command_guards("systemctl --user restart hermes-forge", "local")
    assert res["approved"] is True
