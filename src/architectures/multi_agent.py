"""
Multi-Agent Architecture (Researcher / Critic / Synthesizer).

Three specialized agents connected via an orchestration loop:
  Researcher → retrieves documents
  Critic     → evaluates quality, approves or requests more research
  Synthesizer → produces final cited answer from approved documents
"""
import json
from datetime import datetime

from architectures.common import ResearchResult, extract_citations
from agents.client import FoundryAgentManager
from agents.researcher import create_researcher
from agents.critic import create_critic
from agents.synthesizer import create_synthesizer
from utils import normalize_query, log_query_corrections

__all__ = ["ResearchResult", "MultiAgentOrchestrator"]

MAX_ROUNDS = 3
APPROVAL_THRESHOLD = 0.7


class MultiAgentOrchestrator:
    """Researcher → Critic → Synthesizer pipeline with feedback loop."""

    def __init__(self, manager: FoundryAgentManager, data_sources=None):
        self.manager = manager
        self.researcher, self.tools, self.stats = create_researcher(manager)
        self.critic = create_critic(manager)
        self.synthesizer = create_synthesizer(manager)

    def research(self, query: str, max_results_per_source: int = 5) -> ResearchResult:
        start_time = datetime.now()

        query, corrections = normalize_query(query)
        log_query_corrections(corrections)

        research_output = ""
        rounds = 0
        gaps_feedback = ""

        self.stats.reset()

        # Researcher → Critic feedback loop
        for rnd in range(MAX_ROUNDS):
            rounds = rnd + 1
            prompt = query
            if gaps_feedback:
                prompt += f"\n\nFocus on these gaps:\n{gaps_feedback}"

            research_result = self.manager.run_agent(
                self.researcher.id, prompt, tool_set=self.tools
            )
            research_output = research_result.text

            eval_prompt = (
                f"Research query: {query}\n\n"
                f"Retrieved documents and findings:\n{research_output}"
            )
            eval_result = self.manager.run_agent(self.critic.id, eval_prompt)

            try:
                eval_data = json.loads(eval_result.text)
                score = float(eval_data.get("quality_score", 0))
                is_sufficient = eval_data.get("is_sufficient", False)
                gaps = eval_data.get("gaps", [])
                suggestions = eval_data.get("suggestions", [])
            except (json.JSONDecodeError, ValueError):
                break

            if score >= APPROVAL_THRESHOLD or is_sufficient:
                break

            gaps_feedback = (
                f"Score: {score:.2f}. "
                f"Gaps: {', '.join(gaps)}. "
                f"Suggestions: {', '.join(suggestions)}"
            )

        # Synthesizer produces final answer
        synth_prompt = (
            f"Research query: {query}\n\n"
            f"Approved research findings:\n{research_output}"
        )
        synth_result = self.manager.run_agent(self.synthesizer.id, synth_prompt)
        answer = synth_result.text
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
                "architecture": "multi_agent",
                "rounds": rounds,
            },
        )
