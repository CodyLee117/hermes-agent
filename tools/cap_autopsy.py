"""OMNI-200 S2b (DDR-0036 P-6): cap-out work-preservation + dispatch autopsy.

When a crew agent exhausts its iteration budget with a dispatch still `working`, that
dispatch would otherwise strand until the portal's 3-day janitor (failure mode #8 in
DDR-0036). This best-effort fail-safe runs at the loop's cap-exit and:

  1. finds the agent's OWN working dispatches (OMNI-198 agent-scoped lookup — a Bearer
     token only ever sees its own rows, so no dispatch id needs threading through the
     loop);
  2. preserves uncommitted TRACKED work to a *pushed* recovery ref WITHOUT switching
     HEAD — the crew share ONE physical clone, so a `checkout` would yank HEAD out from
     under a peer mid-work; `git stash create` makes a dangling commit we can push by
     sha. (Untracked files have no HEAD-safe capture, so they're not included — the
     autopsy note says so.)
  3. files the OMNI-200 S1 autopsy (`reason: iteration_cap` + `recovery_ref`) so the
     dispatch ends terminal `dead` with a recovery pointer.

NEVER raises — a fail-safe must not crash the agent's own shutdown path. Every network /
git operation is wrapped and degrades to "filed without recovery_ref" or a no-op.
"""

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 10
_GIT_TIMEOUT = 30


def _api(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode() or "{}")


def _preserve_work(cwd: str, branch: str) -> str | None:
    """Snapshot uncommitted TRACKED changes to a pushed recovery ref WITHOUT a checkout
    (HEAD-safe in the shared clone). Returns the ref name on success, else None."""
    def git(*a):
        return subprocess.run(["git", "-C", cwd, *a],
                              capture_output=True, text=True, timeout=_GIT_TIMEOUT)
    try:
        if git("rev-parse", "--is-inside-work-tree").returncode != 0:
            return None
        snap = git("stash", "create").stdout.strip()
        if not snap:
            return None  # clean tree — nothing to preserve
        push = git("push", "origin", f"{snap}:refs/heads/{branch}")
        if push.returncode != 0:
            logger.warning("cap_autopsy: recovery push failed: %s", (push.stderr or "")[:200])
            return None
        return branch
    except Exception as e:  # subprocess timeout, git missing, etc. — never fatal
        logger.warning("cap_autopsy: work-preservation error: %s", e)
        return None


def file_cap_autopsy(agent_name: str, base_url: str, token: str, cwd: str,
                     max_dispatches: int = 5) -> list[int]:
    """Autopsy the agent's own `working` dispatches after an iteration-cap. Best-effort;
    returns the dispatch ids autopsied (for tests/observability). Never raises."""
    base = (base_url or "").rstrip("/")
    if not (base and agent_name and token):
        return []
    try:
        # OMNI-198: a Bearer token is scoped to its OWN dispatches.
        resp = _api("GET",
                    f"{base}/api/workspace/dispatches?state=working&limit={max_dispatches}",
                    token)
    except Exception as e:
        logger.warning("cap_autopsy: dispatch lookup failed: %s", e)
        return []
    autopsied: list[int] = []
    for d in (resp or {}).get("dispatches", []) or []:
        did = d.get("id")
        if did is None:
            continue
        branch = _preserve_work(cwd, f"recovery/{agent_name}-d{did}-{int(time.time())}")
        note = ("agent hit its iteration cap with this dispatch still working; "
                "auto-filed by the harness fail-safe (OMNI-200 S2b). "
                + (f"uncommitted tracked work preserved at {branch}."
                   if branch else
                   "no recoverable tracked changes (untracked files not captured)."))
        try:
            _api("POST", f"{base}/api/workspace/dispatches/{did}/autopsy", token,
                 {"reason": "iteration_cap", "note": note, "recovery_ref": branch})
            autopsied.append(did)
            logger.info("cap_autopsy: filed autopsy on dispatch %s (recovery=%s)", did, branch)
        except urllib.error.HTTPError as e:
            # 409 = dispatch completed between the cap and the autopsy — benign.
            if e.code != 409:
                logger.warning("cap_autopsy: autopsy POST failed on %s: HTTP %s", did, e.code)
        except Exception as e:
            logger.warning("cap_autopsy: autopsy POST error on %s: %s", did, e)
    return autopsied
