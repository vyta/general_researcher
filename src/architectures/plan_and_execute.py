"""
Plan-and-Execute Architecture.

Two-phase approach:
  Phase 1 — Planner agent creates a step-by-step research plan
  Phase 2 — Researcher agent executes the plan using search tools
  Phase 3 — Critic agent verifies the answer
  Phase 4 — Synthesizer agent produces the final cited response

If execution fails, the Planner is asked to create a revised plan.
"""
import json
from datetime import datetime

from architectures.common import ResearchResult, extract_citations
from agents.client import FoundryAgentManager
from agents.planner import create_planner
from agents.researcher import create_researcher
from agents.critic import create_critic
from agents.synthesizer import create_synthesizer
from utils import normalize_query, log_query_corrections

__all__ = ["ResearchResult", "PlanAndExecuteOrchestrator"]

MAX_REPLANS = 2


class PlanAndExecuteOrchestrator:
    """Planner → Researcher → Critic → Synthesizer pipeline."""

    def __init__(self, manager: FoundryAgentManager, data_sources=None):
        self.manager = manager
        self.planner = create_planner(manager)
        self.researcher, self.tools, self.stats = create_researcher(manager)
        self.critic = create_critic(manager)
        self.synthesizer = create_synthesizer(manager)

    def research(self, query: str, max_results_per_source: int = 5) -> ResearchResult:
        start_time = datetime.now()

        query, corrections = normalize_query(query)
        log_query_corrections(corrections)

        plan_text = ""
        research_output = ""
        replans = 0

        self.stats.reset()

        # Phase 1: Create plan
        plan_text = self.manager.run_agent(self.planner.id, query).text

        # Phase 2: Execute plan — researcher follows the plan
        for attempt in range(MAX_REPLANS + 1):
            exec_prompt = (
                f"Research query: {query}\n\n"
                f"Follow this research plan:\n{plan_text}\n\n"
                f"Execute each step, searching the appropriate sources."
            )
            research_result = self.manager.run_agent(
                self.researcher.id, exec_prompt, tool_set=self.tools
            )
            research_output = research_result.text

            # Phase 3: Critic verifies
            verify_prompt = (
                f"Research query: {query}\n\n"
                f"Research findings:\n{research_output}"
            )
            verify_result = self.manager.run_agent(self.critic.id, verify_prompt)

            try:
                verify_data = json.loads(verify_result.text)
                is_sufficient = verify_data.get("is_sufficient", True)
                score = float(verify_data.get("quality_score", 1.0))
            except (json.JSONDecodeError, ValueError):
                break  # Can't parse verification, proceed with current results

            if is_sufficient or score >= 0.7:
                break

            if attempt < MAX_REPLANS:
                replans += 1
                replan_prompt = (
                    f"Original query: {query}\n\n"
                    f"Previous plan:\n{plan_text}\n\n"
                    f"Critic feedback:\n{verify_result.text}\n\n"
                    f"Create a revised plan to address the gaps."
                )
                plan_text = self.manager.run_agent(self.planner.id, replan_prompt).text

        # Phase 4: Synthesizer produces final answer
        synth_prompt = (
            f"Research query: {query}\n\n"
            f"Research findings:\n{research_output}"
        )
        answer = self.manager.run_agent(self.synthesizer.id, synth_prompt).text
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
                "architecture": "plan_and_execute",
                "replans": replans,
            },
        )
