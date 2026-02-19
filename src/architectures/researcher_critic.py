"""
Researcher-Critic Loop Architecture.

Uses a Researcher agent for document retrieval and a Critic agent for
iterative quality improvement.  The Researcher runs, the Critic evaluates,
and if quality is insufficient the Researcher is re-run with feedback.
"""
import json
from datetime import datetime

from architectures.common import ResearchResult, extract_citations
from agents.client import FoundryAgentManager
from agents.researcher import create_researcher
from agents.critic import create_critic
from utils import normalize_query, log_query_corrections

__all__ = ["ResearchResult", "ResearcherCriticOrchestrator"]

QUALITY_THRESHOLD = 0.7
MAX_ITERATIONS = 3


class ResearcherCriticOrchestrator:
    """Researcher + Critic loop for iterative quality improvement."""

    def __init__(self, manager: FoundryAgentManager, data_sources=None):
        self.manager = manager
        self.agent, self.tools, self.stats = create_researcher(manager)
        self.critic = create_critic(manager)

    def research(self, query: str, max_results_per_source: int = 5) -> ResearchResult:
        start_time = datetime.now()

        query, corrections = normalize_query(query)
        log_query_corrections(corrections)

        answer = ""
        critique_feedback = ""
        iterations = 0
        docs_retrieved = 0
        sources_called = []

        self.stats.reset()

        for iteration in range(MAX_ITERATIONS):
            iterations = iteration + 1
            prompt = query
            if critique_feedback:
                prompt += f"\n\nPrevious feedback to address:\n{critique_feedback}"

            result = self.manager.run_agent(self.agent.id, prompt, tool_set=self.tools)
            answer = result.text

            # Ask critic to evaluate
            eval_prompt = (
                f"Research query: {query}\n\n"
                f"Synthesized answer:\n{answer}"
            )
            eval_result = self.manager.run_agent(self.critic.id, eval_prompt)

            try:
                eval_data = json.loads(eval_result.text)
                score = float(eval_data.get("quality_score", 0))
                is_sufficient = eval_data.get("is_sufficient", False)
                gaps = eval_data.get("gaps", [])
                suggestions = eval_data.get("suggestions", [])
            except (json.JSONDecodeError, ValueError):
                break  # Can't parse critique, use current answer

            if score >= QUALITY_THRESHOLD or is_sufficient:
                break

            critique_feedback = (
                f"Quality score: {score:.2f}. "
                f"Gaps: {', '.join(gaps)}. "
                f"Suggestions: {', '.join(suggestions)}"
            )

        citations = extract_citations(answer)
        docs_retrieved, sources_called = self.stats.reset()

        elapsed = (datetime.now() - start_time).total_seconds()

        return ResearchResult(
            query=query,
            answer=answer,
            sources_checked=sources_called,
            documents_retrieved=docs_retrieved,
            documents_used=len(citations),
            citations=citations,
            time_elapsed=elapsed,
            metadata={
                "architecture": "researcher_critic",
                "iterations": iterations,
            },
        )
