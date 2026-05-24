from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter, _leading_mentions  # noqa: E402


def test_leading_mentions_extraction():
    assert _leading_mentions("<@123> hello") == "<@123>"
    assert _leading_mentions("<@!123> <@456> hi") == "<@!123> <@456>"
    assert _leading_mentions("  <@123>  rest") == "<@123>"
    assert _leading_mentions("no mention here") == ""
    assert _leading_mentions("text <@123> mid-message") == ""  # only leading counts
    assert _leading_mentions("") == ""


@pytest.mark.asyncio
async def test_split_message_reinjects_mention_into_every_chunk():
    """A long message led by a mention must keep that mention on every chunk —
    otherwise only chunk 0 pings the recipient and their monitor misses the rest."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    sent = []

    async def fake_send(*, content, reference=None):
        sent.append(content)
        return SimpleNamespace(id=len(sent))

    channel = SimpleNamespace(
        fetch_message=AsyncMock(),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    mention = "<@1482601532296921158>"
    long_body = " ".join(f"word{i}" for i in range(700))  # well over 2000 chars
    result = await adapter.send("555", f"{mention} {long_body}")

    assert result.success is True
    assert channel.send.await_count >= 2, "message should have split into multiple chunks"
    # Every chunk carries the mention, and stays under the Discord limit.
    for chunk in sent:
        assert mention in chunk, f"chunk missing mention: {chunk[:60]!r}…"
        assert len(chunk) <= adapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_system_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    ref_msg = SimpleNamespace(id=99)
    sent_msg = SimpleNamespace(id=1234)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return sent_msg

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "1234"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 2
    assert send_calls[0]["reference"] is ref_msg
    assert send_calls[1]["reference"] is None
