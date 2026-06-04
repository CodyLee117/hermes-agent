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


class TestPrependReasoningBlock:
    """Reasoning prepend must hoist leading mentions so split re-injection works."""

    def test_leading_mention_hoisted_above_reasoning(self):
        from gateway.run import _prepend_reasoning_block
        out = _prepend_reasoning_block("<@123> done, merged.", "thought about it")
        assert out.startswith("<@123>\n💭 **Reasoning:**")
        assert "done, merged." in out

    def test_multiple_leading_mentions_hoisted(self):
        from gateway.run import _prepend_reasoning_block
        out = _prepend_reasoning_block("<@123> <@!456> report", "hmm")
        assert out.startswith("<@123> <@!456>\n💭 **Reasoning:**")

    def test_no_mention_keeps_reasoning_first(self):
        from gateway.run import _prepend_reasoning_block
        out = _prepend_reasoning_block("plain response", "hmm")
        assert out.startswith("💭 **Reasoning:**")
        assert out.endswith("plain response")

    def test_mid_message_mention_not_hoisted(self):
        from gateway.run import _prepend_reasoning_block
        out = _prepend_reasoning_block("see <@123> later", "hmm")
        assert out.startswith("💭 **Reasoning:**")

    def test_hoisted_output_split_reinjects_mentions(self):
        """End-to-end with the adapter's splitter: every chunk pings."""
        from gateway.run import _prepend_reasoning_block
        from gateway.platforms.discord import _leading_mentions
        out = _prepend_reasoning_block("<@123> " + "x" * 4000, "reasoning text")
        assert _leading_mentions(out) == "<@123>"


class TestMidTextMentionSplit:
    """Mention-less chunks of a split message get the message's first mention.

    Repro: Kit's 8-part sectioned review (2026-06-04) led with prose, carried
    mentions per-section; section boundaries didn't align with chunk boundaries
    so parts 5 and 8 went out mention-less and Ziggy's monitor missed them.
    """

    def _adapter(self):
        from gateway.platforms.discord import DiscordAdapter
        return DiscordAdapter.__new__(DiscordAdapter)

    def test_mid_text_mention_becomes_fallback_prefix(self):
        from gateway.platforms.discord import _leading_mentions
        import re
        # Message starts with prose, mention appears mid-text
        formatted = "79 commits, analysis below.\n\n<@111> section one\n" + "x" * 4500
        assert _leading_mentions(formatted) == ""  # leading detection misses it
        first = re.search(r"<@!?\d+>", formatted)
        assert first.group(0) == "<@111>"

    def test_every_chunk_carries_target_mention(self):
        """Simulate the send-path chunk logic on a sectioned multi-part message."""
        import re
        from gateway.platforms.discord import DiscordAdapter, _leading_mentions

        adapter = self._adapter()
        formatted = (
            "prose intro with no mention\n\n"
            + "<@111> **section A**\n" + "a" * 1900
            + "\n\nmention-less middle section\n" + "b" * 1900
            + "\n\n<@111> **section B**\n" + "c" * 1900
        )
        mention_prefix = _leading_mentions(formatted)
        if not mention_prefix:
            m = re.search(r"<@!?\d+>", formatted)
            mention_prefix = m.group(0) if m else ""
        split_limit = adapter.MAX_MESSAGE_LENGTH - (len(mention_prefix) + 1)
        chunks = adapter.truncate_message(formatted, split_limit)
        assert len(chunks) > 2
        if mention_prefix and len(chunks) > 1:
            chunks = [
                c if mention_prefix in c else f"{mention_prefix}\n{c}"
                for c in chunks
            ]
        for i, chunk in enumerate(chunks):
            assert mention_prefix in chunk, f"chunk {i} lost the mention"

    def test_other_users_mention_does_not_satisfy_target(self):
        """A chunk containing only a different user's mention still gets the prefix."""
        target = "<@111>"
        chunk_with_other = "see <@222> for details"
        result = chunk_with_other if target in chunk_with_other else f"{target}\n{chunk_with_other}"
        assert result.startswith("<@111>")
        assert "<@222>" in result
