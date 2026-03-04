"""
ACP Agent Architecture.

Wraps a local ACP agent as a research architecture so the eval
framework can test it identically to Foundry-based architectures.

The ACP agent is expected to handle the full workflow internally
(search, reason, synthesize).  Its tool calls and internal state
are opaque — the framework only sees the final textual answer.
"""
from datetime import datetime
from typing import Optional

from architectures.common import ResearchResult, extract_citations
from agents.acp_client import ACPClient

__all__ = ["ACPAgentOrchestrator"]


class ACPAgentOrchestrator:
    """Orchestrator that delegates to an ACP agent.

    Conforms to the same interface as other orchestrators so that
    ``eval.py``, ``main.py``, and the action/step framework work
    unchanged:

    * Constructor accepts ``(manager, data_sources)`` — both are
      ignored because the ACP agent manages its own tools.
    * ``.research(query)`` returns a :class:`ResearchResult`.

    Connection to the ACP server is established lazily on the first
    call to :meth:`research` and torn down via :meth:`close`.
    """

    def __init__(self, manager=None, data_sources=None,
                 acp_config=None):
        """
        Args:
            manager:      Ignored (kept for interface compat).
            data_sources: Ignored (kept for interface compat).
            acp_config:   :class:`~agents.smart_inventory_advisor.ACPAgentConfig`.
                          Defaults to ``SMART_INVENTORY_ADVISOR``.
        """
        from agents.smart_inventory_advisor import SMART_INVENTORY_ADVISOR
        self.config = acp_config or SMART_INVENTORY_ADVISOR
        self._client: Optional[ACPClient] = None
        self._session_id: Optional[str] = None

    # ── Lazy connection ───────────────────────────────────────────────

    def _ensure_connected(self):
        if self._client is not None:
            return
        self._client = ACPClient(self.config)
        self._client.connect()
        self._session_id = self._client.new_session(cwd=self.config.cwd)

    # ── Public API ────────────────────────────────────────────────────

    def research(self, query: str,
                 max_results_per_source: int = 5) -> ResearchResult:
        """Send *query* to the ACP agent and return a ``ResearchResult``."""
        self._ensure_connected()
        start_time = datetime.now()

        result = self._client.prompt(query, session_id=self._session_id)

        elapsed = (datetime.now() - start_time).total_seconds()
        citations = extract_citations(result.text)

        return ResearchResult(
            query=query,
            answer=result.text,
            sources_checked=[],           # opaque — agent handles its own
            documents_retrieved=0,         # opaque
            documents_used=len(citations),
            citations=citations,
            time_elapsed=elapsed,
            metadata={
                "architecture": "acp_agent",
                "agent_name": self.config.name,
            },
        )

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        """Disconnect from the ACP agent."""
        if self._client:
            self._client.close()
            self._client = None
            self._session_id = None

    def __del__(self):
        self.close()
