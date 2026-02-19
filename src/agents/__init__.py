"""
Shared Foundry Agent definitions for the General Researcher.

This package provides 5 reusable agent roles that architectures compose
in different patterns:

    Researcher      — searches all data sources (has tools)
    Critic          — evaluates quality, identifies gaps (no tools)
    Synthesizer     — creates cited answers from documents (no tools)
    Planner         — creates step-by-step research plans (no tools)
    Source Worker   — searches one specific data source (has one tool)

Each architecture in src/architectures/ is a thin orchestration layer
that creates and invokes these agents in different combinations.
"""
from agents.client import FoundryAgentManager
from agents.researcher import create_researcher, RESEARCHER_INSTRUCTIONS
from agents.critic import create_critic, CRITIC_INSTRUCTIONS
from agents.synthesizer import create_synthesizer, SYNTHESIZER_INSTRUCTIONS
from agents.planner import create_planner, PLANNER_INSTRUCTIONS
from agents.source_worker import create_source_worker, SOURCE_WORKER_INSTRUCTIONS

__all__ = [
    "FoundryAgentManager",
    "create_researcher",
    "create_critic",
    "create_synthesizer",
    "create_planner",
    "create_source_worker",
    "RESEARCHER_INSTRUCTIONS",
    "CRITIC_INSTRUCTIONS",
    "SYNTHESIZER_INSTRUCTIONS",
    "PLANNER_INSTRUCTIONS",
    "SOURCE_WORKER_INSTRUCTIONS",
]
