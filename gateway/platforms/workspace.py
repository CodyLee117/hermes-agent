"""Workspace chat platform adapter (Omnipotence OMNI-058b, DDR-0015 W3b).

Native workspace chat becomes a first-class Hermes platform, exactly like
Discord is today: the adapter subscribes OUTBOUND to the portal's SSE inbox
(agent hosts are unreachable from the portal — DDR-0014 D1) and replies via
the machine POST path. The Bearer token names the agent; the portal decides
the principal, so this adapter can never speak as anyone else.

Config (env, per-profile .env):
  WORKSPACE_CHAT_URL    portal base, e.g. https://agents.cocolee.co
  WORKSPACE_CHAT_AGENT  this agent's name, e.g. kit
  WORKSPACE_CHAT_TOKEN  this agent's chat_token_<agent> value

Dispatch policy (loop safety with a multi-agent commons): DM channels always
dispatch; non-DM channels dispatch only when the body mentions this agent's
name (the requireMention posture). The portal inbox already excludes the
agent's own messages server-side; we re-check here as a second rail.

Graceful-dark (DDR-0015 kill rail): the listen loop reconnects forever with
backoff and never takes the gateway down; a dead leg means workspace chat
goes quiet for this agent while Discord keeps working.
"""
import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
import logging

logger = logging.getLogger(__name__)

BACKOFF_START_SEC = 2.0
BACKOFF_MAX_SEC = 120.0
# Portal-side BODY_MAX_CHARS is 8000; keep headroom for chunk indicators.
MAX_MESSAGE_LENGTH = 7800


def check_workspace_requirements() -> bool:
    """aiohttp ships with the gateway; the real requirement is config."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("WORKSPACE_CHAT_URL") and os.getenv("WORKSPACE_CHAT_AGENT")
                and os.getenv("WORKSPACE_CHAT_TOKEN"))


class WorkspaceAdapter(BasePlatformAdapter):
    """SSE-subscribe inbox + machine-POST replies against the agents portal."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WORKSPACE)
        self.base_url = (os.getenv("WORKSPACE_CHAT_URL") or "").rstrip("/")
        self.agent = os.getenv("WORKSPACE_CHAT_AGENT") or ""
        self.token = os.getenv("WORKSPACE_CHAT_TOKEN") or ""
        self._channels: Dict[str, dict] = {}  # chat_id -> channel meta (from hello)
        self._listen_task: Optional[asyncio.Task] = None
        self._http = None  # aiohttp.ClientSession, created on connect
        self._mention_re = re.compile(rf"\b{re.escape(self.agent)}\b", re.IGNORECASE) if self.agent else None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        import aiohttp
        if not (self.base_url and self.agent and self.token):
            logger.warning("workspace: WORKSPACE_CHAT_URL/AGENT/TOKEN not configured")
            return False
        self._http = aiohttp.ClientSession()
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._mark_connected()
        logger.info(f"workspace: adapter up for agent '{self.agent}' -> {self.base_url}")
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self._http:
            await self._http.close()
            self._http = None

    # ── inbox (SSE) ──────────────────────────────────────────────────────────

    async def _listen_loop(self) -> None:
        backoff = BACKOFF_START_SEC
        while True:
            started = asyncio.get_event_loop().time()
            try:
                await self._subscribe_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(f"workspace: stream dropped: {exc}")
            lived = asyncio.get_event_loop().time() - started
            backoff = BACKOFF_START_SEC if lived > 60 else min(backoff * 2, BACKOFF_MAX_SEC)
            await asyncio.sleep(backoff)

    async def _subscribe_once(self) -> None:
        import aiohttp
        url = f"{self.base_url}/api/chat/inbox?agent={self.agent}"
        timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
        async with self._http.get(
            url, timeout=timeout,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "text/event-stream"},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"inbox HTTP {resp.status}")
            event, data_lines = None, []
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith(":"):
                    continue
                if line == "":
                    if event and data_lines:
                        try:
                            await self._on_frame(event, json.loads("\n".join(data_lines)))
                        except ValueError:
                            pass
                    event, data_lines = None, []
                elif line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data_lines.append(line[6:])

    async def _on_frame(self, event: str, data: dict) -> None:
        if event == "inbox.hello":
            self._channels = {str(c["id"]): c for c in data.get("channels", [])}
            logger.info(f"workspace: connected as {self.agent}, {len(self._channels)} channel(s)")
            return
        if event != "chat.message":
            return
        msg = data.get("message", {})
        chat_id = str(data.get("channel_id", ""))
        principal = str(msg.get("principal", ""))
        body = str(msg.get("body", ""))
        if not chat_id or not body or principal == self.agent:
            return  # second self-echo rail; server already excludes own messages
        chan = self._channels.get(chat_id, {})
        is_dm = chan.get("kind") == "dm"
        if not is_dm and self._mention_re and not self._mention_re.search(body):
            return  # requireMention posture in shared channels — loop safety
        source = self.build_source(
            chat_id=chat_id,
            chat_name=f"#{chan.get('name', chat_id)}",
            chat_type="dm" if is_dm else "group",
            user_id=principal,
            user_name=principal,
        )
        await self.handle_message(MessageEvent(
            text=body,
            message_type=MessageType.TEXT,
            source=source,
            message_id=str(msg.get("id", "")),
            raw_message=data,
        ))

    # ── outbound ─────────────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str,
                   reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if not self._http:
            return SendResult(success=False, error="not connected", retryable=True)
        try:
            async with self._http.post(
                f"{self.base_url}/api/chat/channels/{chat_id}/messages",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"body": content},
            ) as resp:
                payload = await resp.json(content_type=None)
                if resp.status == 200:
                    return SendResult(success=True,
                                      message_id=str(payload.get("message", {}).get("id", "")),
                                      raw_response=payload)
                detail = payload.get("detail", f"HTTP {resp.status}")
                return SendResult(success=False, error=str(detail),
                                  retryable=(resp.status in (429, 502, 503)))
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chan = self._channels.get(str(chat_id), {})
        return {
            "name": chan.get("name", chat_id),
            "type": "dm" if chan.get("kind") == "dm" else "group",
            "chat_id": str(chat_id),
        }
