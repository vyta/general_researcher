"""
Hybrid P2P + Hierarchical Architecture.

Source Workers search in parallel, share discoveries between rounds,
and a Synthesizer produces the final answer.  The orchestrator handles
discovery injection (simulating peer-to-peer communication).
"""
import json
from datetime import datetime
from typing import List

from data_sources.base import DataSource
from architectures.common import ResearchResult, extract_citations
from agents.client import FoundryAgentManager
from agents.source_worker import create_source_worker
from agents.synthesizer import create_synthesizer
from utils import normalize_query, log_query_corrections

__all__ = ["ResearchResult", "HybridP2PHierarchical"]

MAX_ROUNDS = 2


class HybridP2PHierarchical:
    """Source Workers with peer discovery sharing → Synthesizer."""

    def __init__(self, manager: FoundryAgentManager, data_sources: List[DataSource] = None):
        self.manager = manager
        self.data_sources = data_sources or []
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

        all_findings = {}
        all_discoveries = {}
        for s in self._all_stats:
            s.reset()

        for rnd in range(MAX_ROUNDS):
            # Build peer context from previous round discoveries
            peer_context = ""
            if all_discoveries:
                peer_context = "\n\nDiscoveries from other sources:\n"
                for src_name, discoveries in all_discoveries.items():
                    peer_context += f"- {src_name}: {', '.join(discoveries)}\n"

            # Run each worker with peer context
            for worker_agent, worker_tools, source_name in self.workers:
                worker_prompt = f"Research query: {query}"
                if peer_context:
                    other_discoveries = {
                        k: v for k, v in all_discoveries.items()
                        if k != source_name
                    }
                    if other_discoveries:
                        worker_prompt += (
                            "\n\nDiscoveries from peer researchers:\n"
                            + "\n".join(
                                f"- {k}: {', '.join(v)}"
                                for k, v in other_discoveries.items()
                            )
                            + "\n\nUse these to refine your search."
                        )

                worker_result = self.manager.run_agent(
                    worker_agent.id, worker_prompt, tool_set=worker_tools
                )
                all_findings[source_name] = worker_result.text

                # Extract discoveries for peer sharing
                try:
                    data = json.loads(worker_result.text)
                    discoveries = data.get("key_discoveries", [])
                except (json.JSONDecodeError, ValueError):
                    discoveries = []
                all_discoveries[source_name] = discoveries

        # Synthesizer produces final answer from all findings
        findings_text = "\n\n".join(
            f"--- {name} ---\n{text}"
            for name, text in all_findings.items()
        )
        synth_prompt = (
            f"Research query: {query}\n\n"
            f"Research findings from all sources:\n{findings_text}"
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
                "architecture": "hybrid_p2p",
                "rounds": MAX_ROUNDS,
                "worker_count": len(self.workers),
            },
        )
