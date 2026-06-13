"""Per-channel project-context injection for the gateway.

Mirrors Ziggy's ``UserPromptSubmit`` channel→project-load hook on the Claude
Code side: when an inbound message arrives on a chat/channel that
``channel-map.json`` maps to a project, that project's rules/workflow doc is
appended to the ephemeral system prompt so the agent auto-onboards to the
project every session — no manual hand-off needed.

Resolution order (per-agent HERMES_HOME first, then the shared ~/.hermes):

    {HERMES_HOME}/channel-map.json            → ~/.hermes/channel-map.json
    {HERMES_HOME}/project-rules/<project>.md  → ~/.hermes/project-rules/<project>.md

Map shape (mirrors ``~/.ziggy/workspace/.claude/channel-map.json``)::

    {
      "channels":      {"<chat_id>": "<project>"},
      "skip_channels": {"<chat_id>": "why this channel is intentionally unmapped"}
    }

Semantics:

- **Additive per session** — sessions are user-keyed
  (``group_sessions_per_user``) and can span channels; once a project loads
  for a session it stays loaded, guarded on (session_id × project).
- **Idempotent** — the ephemeral system prompt is rebuilt every message and
  is never saved to trajectories, so the in-memory state only controls the
  "loaded" log line; losing it on gateway restart is harmless.
- **Fail-open** — any error (missing map, bad JSON, missing doc) degrades to
  a no-op so message handling is never blocked.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# OMNI-079b: the operating-procedures loader. Fetch this agent's resolved
# rules-as-data from the portal and inject them, generalizing the project-doc
# injection above. Cached briefly so per-message injection isn't an HTTP call.
_proc_cache: Dict[tuple, Tuple[float, str]] = {}
_PROC_TTL = 60.0

# Safety cap per project doc so a runaway rules file can't blow up the prompt.
_MAX_DOC_CHARS = 32_768

# Only sane project slugs may resolve to a file (guards the map against
# path-traversal entries like "../../etc/passwd").
_PROJECT_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# path → (mtime, text) cache so per-message lookups stay a stat(), not a read.
_file_cache: Dict[str, Tuple[float, str]] = {}

# session_id → projects already loaded for that session (additive, ordered).
_session_projects: Dict[str, List[str]] = {}

# Backstop against unbounded growth on a very long-lived gateway.  Clearing
# only costs a repeated "loaded" log line — injection itself is stateless.
_MAX_TRACKED_SESSIONS = 4096


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _candidate_paths(rel: str) -> List[Path]:
    """Per-agent HERMES_HOME first, shared ~/.hermes as fallback (deduped)."""
    candidates = [_hermes_home() / rel, Path.home() / ".hermes" / rel]
    seen, out = set(), []
    for path in candidates:
        if str(path) not in seen:
            seen.add(str(path))
            out.append(path)
    return out


def _read_cached(path: Path) -> Optional[str]:
    """Read *path* through the mtime cache; None when unreadable/missing."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    hit = _file_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("project-context: cannot read %s: %s", path, exc)
        return None
    _file_cache[key] = (mtime, text)
    return text


def _load_channel_map() -> Dict[str, str]:
    """Return {chat_id: project} from the first resolvable channel map."""
    for path in _candidate_paths("channel-map.json"):
        text = _read_cached(path)
        if text is None:
            continue
        try:
            data = json.loads(text)
        except ValueError as exc:
            logger.warning("project-context: invalid JSON in %s: %s", path, exc)
            return {}
        channels = data.get("channels") if isinstance(data, dict) else None
        if isinstance(channels, dict):
            return {str(k): str(v) for k, v in channels.items()}
        return {}
    return {}


def _load_project_doc(project: str) -> Optional[str]:
    """Return the rules doc for *project*, or None when absent."""
    if not _PROJECT_SLUG_RE.fullmatch(project):
        logger.warning("project-context: ignoring unsafe project slug %r", project)
        return None
    for path in _candidate_paths(f"project-rules/{project}.md"):
        text = _read_cached(path)
        if text is not None:
            if len(text) > _MAX_DOC_CHARS:
                logger.warning(
                    "project-context: %s truncated to %d chars", path, _MAX_DOC_CHARS,
                )
                text = text[:_MAX_DOC_CHARS]
            return text
    logger.debug("project-context: no rules doc found for %r", project)
    return None


def _load_procedures_block(projects: List[str]) -> str:
    """OMNI-079b: the agent's resolved operating procedures (global + type +
    project + agent), fetched from the portal. Best-effort, cached, fail-open —
    a portal hiccup must never block message handling. Loads even with no
    project mapped (global rules always apply)."""
    base = (os.getenv("WORKSPACE_CHAT_URL") or "").rstrip("/")
    token = os.getenv("WORKSPACE_CHAT_TOKEN") or ""
    agent = os.getenv("WORKSPACE_CHAT_AGENT") or ""
    if not (base and token and agent):
        return ""
    project = projects[0] if projects else None
    key = (agent, project)
    now = time.time()
    hit = _proc_cache.get(key)
    if hit and hit[0] > now:
        return hit[1]
    params = {"agent": agent, "type": "hermes"}
    if project:
        params["project"] = project
    url = f"{base}/api/workspace/procedures/resolved?{urllib.parse.urlencode(params)}"
    block = ""
    try:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}",
                          "User-Agent": "hermes-procedures/0.1"})
        with urllib.request.urlopen(req, timeout=3) as r:
            procs = json.loads(r.read() or b"{}").get("procedures", [])
        if procs:
            items = "\n\n".join(f"**{p['title']}**\n{p['body']}" for p in procs)
            block = ("## Operating Procedures (auto-loaded)\n\n"
                     "Follow these rules for the duration of this session.\n\n" + items)
    except Exception as exc:  # fail-open
        logger.debug("procedures injection failed (non-fatal): %s", exc)
        return ""  # don't cache a transient failure
    _proc_cache[key] = (now + _PROC_TTL, block)
    return block


def get_project_context_block(source, session_id: str) -> str:
    """Build the project-context block for this message ("" when nothing maps).

    Appended by the gateway to the ephemeral system prompt right after
    ``build_session_context_prompt``.  Never raises.
    """
    try:
        channel_map = _load_channel_map()
        if len(_session_projects) > _MAX_TRACKED_SESSIONS:
            _session_projects.clear()
        loaded = _session_projects.setdefault(str(session_id), [])

        # Match on the chat id and (for platform threads) the thread id.
        for cid in (getattr(source, "chat_id", None), getattr(source, "thread_id", None)):
            project = channel_map.get(str(cid)) if cid else None
            if project and project not in loaded:
                loaded.append(project)
                logger.info(
                    "[Gateway] Project context '%s' loaded for session %s (channel %s)",
                    project, session_id, cid,
                )

        blocks = []
        sections = []
        for project in loaded:
            doc = _load_project_doc(project)
            if doc and doc.strip():
                sections.append(f"### Project: {project}\n\n{doc.strip()}")
        if sections:
            blocks.append(
                "## Project Context (auto-loaded)\n\n"
                "This channel maps to the project(s) below. Follow their rules and "
                "workflow for the duration of this session.\n\n"
                + "\n\n---\n\n".join(sections)
            )

        # OMNI-079b: operating procedures load regardless of project mapping
        # (global rules always apply); project-scoped ones use the loaded project.
        proc_block = _load_procedures_block(loaded)
        if proc_block:
            blocks.append(proc_block)

        return "\n\n---\n\n".join(blocks)
    except Exception as exc:  # fail-open: never block message handling
        logger.warning("project-context injection failed (non-fatal): %s", exc)
        return ""


def _reset_state_for_tests() -> None:
    """Clear module caches/state (test helper)."""
    _file_cache.clear()
    _session_projects.clear()
    _proc_cache.clear()
