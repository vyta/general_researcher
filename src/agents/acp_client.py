"""
ACP (Agent Client Protocol) client.

Thin wrapper around the official ``agent-client-protocol`` Python SDK
(https://github.com/agentclientprotocol/python-sdk).

Supports stdio (subprocess) and TCP transport modes.  The SDK is fully
async; this module bridges to the synchronous calling convention used
by the rest of the test framework.

Protocol flow:
    initialize → session/new → session/prompt (with streaming updates)

Reference:
    https://agentclientprotocol.com/protocol/overview
    https://agentclientprotocol.github.io/python-sdk/quickstart/
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from acp import (
    PROTOCOL_VERSION,
    connect_to_agent,
    spawn_agent_process,
    text_block,
)
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    RequestPermissionResponse,
    TextContentBlock,
)

logger = logging.getLogger(__name__)

__all__ = ["ACPClient", "ACPRunResult", "PROTOCOL_VERSION"]


# ═══════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ACPRunResult:
    """Result from an ACP agent prompt."""
    text: str
    documents_retrieved: int = 0
    sources_called: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Client callback handler (agent → client messages)
# ═══════════════════════════════════════════════════════════════════════

class _EvalClient(Client):
    """Minimal ``Client`` following the SDK quickstart pattern.

    Only two callbacks are needed — the SDK handles missing methods
    (filesystem, terminal, extensions) by returning ``method_not_found``
    automatically.

    * ``session_update`` — accumulates agent text chunks.
    * ``request_permission`` — auto-approves so the agent can use tools.
    """

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def clear(self) -> None:
        self.chunks.clear()

    async def request_permission(self, options, session_id, tool_call,
                                 **kwargs: Any) -> RequestPermissionResponse:
        """Auto-approve every permission request during testing."""
        # Pick the first "allow" option, or fall back to the first option
        chosen_id = options[0].option_id if options else "allow_once"
        if options:
            for opt in options:
                kind = getattr(opt, 'kind', '') or ''
                if 'allow' in kind.lower():
                    chosen_id = opt.option_id
                    break
        logger.debug("ACP permission auto-approved: tool=%s option=%s",
                     getattr(tool_call, 'title', '?'), chosen_id)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=chosen_id),
        )

    async def session_update(self, session_id, update, **kwargs: Any):
        """Accumulate text from ``AgentMessageChunk`` updates."""
        logger.debug("ACP update: %s", type(update).__name__)
        if isinstance(update, AgentMessageChunk):
            if isinstance(update.content, TextContentBlock):
                self.chunks.append(update.content.text)


# ═══════════════════════════════════════════════════════════════════════
# High-level synchronous client
# ═══════════════════════════════════════════════════════════════════════

class ACPClient:
    """Synchronous wrapper around the async ACP SDK.

    Usage::

        client.connect()
        sid = client.new_session()
        result = client.prompt("Hello", session_id=sid)
        client.close()

    Or as a context manager::

        with ACPClient(config) as client:
            sid = client.new_session()
            result = client.prompt("Hello", session_id=sid)
    """

    def __init__(self, config) -> None:
        self.config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client = _EvalClient()
        self._conn = None          # ClientSideConnection
        self._process = None       # subprocess (stdio only)
        self._session_id: Optional[str] = None
        self._cleanup_stack = None  # async context manager for spawn

    # ── Connection lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Establish transport and perform the ACP ``initialize`` handshake."""
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_connect())

    async def _async_connect(self) -> None:
        if self.config.transport == "stdio":
            await self._connect_stdio()
        elif self.config.transport == "tcp":
            await self._connect_tcp()
        else:
            raise ValueError(f"Unknown ACP transport: {self.config.transport}")

        await self._conn.initialize(protocol_version=PROTOCOL_VERSION)
        logger.info("ACP initialized (protocol_version=%s)", PROTOCOL_VERSION)

    async def _connect_stdio(self) -> None:
        args = ["--acp", "--stdio"] + self.config.extra_args
        cwd = self.config.cwd or os.getcwd()
        self._cleanup_stack = spawn_agent_process(
            self._client, self.config.executable, *args, cwd=cwd,
        )
        self._conn, self._process = await self._cleanup_stack.__aenter__()

    async def _connect_tcp(self) -> None:
        reader, writer = await asyncio.open_connection(
            self.config.host, self.config.port,
        )
        self._conn = connect_to_agent(self._client, writer, reader)
        logger.info("Connected to ACP server at %s:%d",
                     self.config.host, self.config.port)

    # ── Session management ────────────────────────────────────────────

    def new_session(self, cwd: Optional[str] = None) -> str:
        """Create a new ACP session.  Returns the session ID."""
        resp = self._run(self._conn.new_session(
            cwd=cwd or self.config.cwd or os.getcwd(),
            mcp_servers=[],
        ))
        self._session_id = resp.session_id
        logger.info("ACP session created: %s", self._session_id)
        return self._session_id

    # ── Prompting ─────────────────────────────────────────────────────

    def prompt(self, text: str,
               session_id: Optional[str] = None) -> ACPRunResult:
        """Send a user prompt and block until the agent responds."""
        sid = session_id or self._session_id
        if not sid:
            raise RuntimeError("No active session — call new_session() first")

        self._client.clear()

        async def _do_prompt():
            return await asyncio.wait_for(
                self._conn.prompt(
                    prompt=[text_block(text)],
                    session_id=sid,
                ),
                timeout=self.config.timeout,
            )

        try:
            prompt_response = self._run(_do_prompt())
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"ACP prompt did not complete within {self.config.timeout}s"
            ) from exc

        # Log the full prompt response for debugging
        logger.debug("ACP prompt response: %s", repr(prompt_response)[:500])

        response_text = "".join(self._client.chunks)
        logger.info("ACP prompt complete: %d chars", len(response_text))
        return ACPRunResult(text=response_text)

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Tear down transport and release resources."""
        if self._loop and not self._loop.is_closed():
            try:
                if self._cleanup_stack:
                    self._loop.run_until_complete(
                        self._cleanup_stack.__aexit__(None, None, None)
                    )
                    self._cleanup_stack = None
                elif self._conn:
                    self._loop.run_until_complete(self._conn.close())
            except Exception:
                logger.debug("ACP close error (ignored)", exc_info=True)
            finally:
                self._conn = None
                self._process = None
                self._session_id = None
                self._loop.close()
                self._loop = None

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────

    def _run(self, coro):
        """Run an async coroutine on the dedicated event loop."""
        return self._loop.run_until_complete(coro)
