"""
Single Agent Architecture.

Simplest pattern: a single Foundry Researcher agent with search tools.
The agent's native tool-calling loop handles reasoning and acting.
"""
from datetime import datetime

from architectures.common import ResearchResult, extract_citations
from agents.client import FoundryAgentManager
from agents.researcher import create_researcher
from utils import normalize_query, log_query_corrections

# Re-export for backward compatibility (architectures/__init__.py imports it)
__all__ = ["ResearchResult", "SingleAgentOrchestrator"]


class SingleAgentOrchestrator:
    """Single Researcher agent with search tools — simplest architecture."""

    def __init__(self, manager: FoundryAgentManager, data_sources=None):
        self.manager = manager
        self.agent, self.tools, self.stats = create_researcher(manager)

    def research(self, query: str, max_results_per_source: int = 5) -> ResearchResult:
        """Run the Foundry researcher agent and return a ResearchResult."""
        start_time = datetime.now()

        query, corrections = normalize_query(query)
        log_query_corrections(corrections)

        self.stats.reset()
        result = self.manager.run_agent(self.agent.id, query, tool_set=self.tools)
        docs_retrieved, sources_called = self.stats.reset()
        citations = extract_citations(result.text)

        elapsed = (datetime.now() - start_time).total_seconds()

        return ResearchResult(
            query=query,
            answer=result.text,
            sources_checked=sources_called,
            documents_retrieved=docs_retrieved,
            documents_used=len(citations),
            citations=citations,
            time_elapsed=elapsed,
            metadata={"architecture": "single_agent"},
        )
