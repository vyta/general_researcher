"""Init file for architectures module."""
from architectures.common import ResearchResult

from architectures.researcher_critic import ResearcherCriticOrchestrator
from architectures.multi_agent import MultiAgentOrchestrator
from architectures.supervisor_worker import SupervisorWorkerOrchestrator
from architectures.single_agent import SingleAgentOrchestrator
from architectures.single_agent_code import SingleAgentCodeOrchestrator
from architectures.plan_and_execute import PlanAndExecuteOrchestrator
from architectures.hybrid_p2p import HybridP2PHierarchical
from architectures.acp_agent import ACPAgentOrchestrator

# Canonical registry of all architectures
ARCHITECTURES = {
    "single_agent": {
        "name": "Single Agent",
        "class": SingleAgentOrchestrator,
        "emoji": "🔄",
    },
    "single_agent_code": {
        "name": "Single Agent + Code Execution",
        "class": SingleAgentCodeOrchestrator,
        "emoji": "💻",
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
    "acp_agent": {
        "name": "ACP Agent (Smart Inventory Advisor)",
        "class": ACPAgentOrchestrator,
        "emoji": "🔌",
        "external": True,  # requires a running ACP server
    },
}

__all__ = [
    'ResearchResult',
    'SingleAgentOrchestrator',
    'SingleAgentCodeOrchestrator',
    'ResearcherCriticOrchestrator',
    'MultiAgentOrchestrator', 
    'SupervisorWorkerOrchestrator',
    'PlanAndExecuteOrchestrator',
    'HybridP2PHierarchical',
    'ACPAgentOrchestrator',
    'ARCHITECTURES',
]
