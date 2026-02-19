"""Init file for architectures module."""
from architectures.common import ResearchResult

from architectures.researcher_critic import ResearcherCriticOrchestrator
from architectures.multi_agent import MultiAgentOrchestrator
from architectures.supervisor_worker import SupervisorWorkerOrchestrator
from architectures.single_agent import SingleAgentOrchestrator
from architectures.plan_and_execute import PlanAndExecuteOrchestrator
from architectures.hybrid_p2p import HybridP2PHierarchical

# Canonical registry of all architectures
ARCHITECTURES = {
    "single_agent": {
        "name": "Single Agent",
        "class": SingleAgentOrchestrator,
        "emoji": "🔄",
    },
    "researcher_critic": {
        "name": "Researcher-Critic Loop",
        "class": ResearcherCriticOrchestrator,
        "emoji": "🔁",
    },
    "multi_agent": {
        "name": "Multi-Agent (Researcher/Critic/Synthesizer)",
        "class": MultiAgentOrchestrator,
        "emoji": "🤖",
    },
    "supervisor_worker": {
        "name": "Supervisor-Worker",
        "class": SupervisorWorkerOrchestrator,
        "emoji": "👔",
    },
    "plan_execute": {
        "name": "Plan-and-Execute",
        "class": PlanAndExecuteOrchestrator,
        "emoji": "📝",
    },
    "hybrid_p2p": {
        "name": "Hybrid P2P+Hierarchical",
        "class": HybridP2PHierarchical,
        "emoji": "🔗",
    },
}

__all__ = [
    'ResearchResult',
    'SingleAgentOrchestrator',
    'ResearcherCriticOrchestrator',
    'MultiAgentOrchestrator', 
    'SupervisorWorkerOrchestrator',
    'PlanAndExecuteOrchestrator',
    'HybridP2PHierarchical',
    'ARCHITECTURES',
]
