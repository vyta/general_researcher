"""
Supervisor-Worker Architecture.

A Planner agent creates a research strategy, per-source Source Worker agents
execute searches in parallel, a Critic evaluates results, and a Synthesizer
produces the final cited answer.
"""
import json
from datetime import datetime
from typing import List

from data_sources.base import DataSource
from architectures.common import ResearchResult, extract_citations
from agents.client import FoundryAgentManager
from agents.planner import create_planner
from agents.source_worker import create_source_worker
from agents.critic import create_critic
from agents.synthesizer import create_synthesizer
from utils import normalize_query, log_query_corrections

__all__ = ["ResearchResult", "SupervisorWorkerOrchestrator"]

MAX_ROUNDS = 3


class SupervisorWorkerOrchestrator:
    """Planner → Source Workers (parallel) → Critic → Synthesizer."""

    def __init__(self, manager: FoundryAgentManager, data_sources: List[DataSource] = None):
        self.manager = manager
        self.data_sources = data_sources or []
        self.planner = create_planner(manager)
        self.critic = create_critic(manager)
        self.synthesizer = create_synthesizer(manager)
        # Create per-source workers with stats tracking
        self.workers = []
        self._all_stats = []
        for source in self.data_sources:
            worker_agent, worker_tools, worker_stats = create_source_worker(manager, source)
            self.workers.append((worker_agent, worker_tools, source.name))
            self._all_stats.append(worker_stats)

    def research(self, query: str, max_results_per_source: int = 5) -> ResearchResult:
        start_time = datetime.now()

        query, corrections = normalize_query(query)
        log_query_corrections(corrections)

        all_findings = ""
        rounds = 0
        for s in self._all_stats:
            s.reset()

        for rnd in range(MAX_ROUNDS):
            rounds = rnd + 1

            # Planner creates/refines strategy
            plan_prompt = query
            if all_findings:
                plan_prompt += (
                    f"\n\nPrevious findings:\n{all_findings}\n\n"
                    "Identify gaps and create a revised plan."
                )
            plan_response = self.manager.run_agent(self.planner.id, plan_prompt).text

            # Workers execute in sequence (Foundry agents are synchronous)
            round_findings = []
            for worker_agent, worker_tools, source_name in self.workers:
                worker_prompt = (
                    f"Research query: {query}\n\n"
                    f"Plan guidance:\n{plan_response}"
                )
                worker_result = self.manager.run_agent(
                    worker_agent.id, worker_prompt, tool_set=worker_tools
                )
                round_findings.append(f"--- {source_name} ---\n{worker_result.text}")

            all_findings = "\n\n".join(round_findings)

            # Critic evaluates combined results
            eval_prompt = (
                f"Research query: {query}\n\n"
                f"Combined worker findings:\n{all_findings}"
            )
            eval_result = self.manager.run_agent(self.critic.id, eval_prompt)

            try:
                eval_data = json.loads(eval_result.text)
                is_sufficient = eval_data.get("is_sufficient", True)
                score = float(eval_data.get("quality_score", 1.0))
            except (json.JSONDecodeError, ValueError):
                break

            if is_sufficient or score >= 0.7:
                break

        # Synthesizer produces final answer
        synth_prompt = (
            f"Research query: {query}\n\n"
            f"Research findings from all sources:\n{all_findings}"
        )
        answer = self.manager.run_agent(self.synthesizer.id, synth_prompt).text
        citations = extract_citations(answer)

        # Aggregate stats from all worker stats trackers
        docs_retrieved = 0
        sources_called = []
        for s in self._all_stats:
            d, sc = s.reset()
            docs_retrieved += d
            sources_called.extend(x for x in sc if x not in sources_called)

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
                "architecture": "supervisor_worker",
                "rounds": rounds,
                "worker_count": len(self.workers),
            },
        )
