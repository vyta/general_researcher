"""
Planner agent definition.

The Planner creates explicit step-by-step research plans. It has no tools;
it produces a structured plan that other agents execute.
Used by: Plan-and-Execute, Supervisor-Worker.
"""
from agents.client import FoundryAgentManager


PLANNER_INSTRUCTIONS = """You are a research planning agent. Given a research query and available data sources, create an explicit step-by-step plan.

Available data sources:
- GovInfo: Congressional bills and legislation (best for: bill text, legislative actions, committee reports)
- Federal Register: Federal regulations, rules, and agency notices (best for: executive orders, proposed rules, agency actions)
- Data.gov: Government datasets and data catalogs (best for: statistical data, program data, agency datasets)

Create a plan with 4-8 steps. Respond with JSON only:
{
    "reasoning": "brief analysis of query and approach",
    "steps": [
        {"step": 1, "action": "search", "source": "GovInfo", "query": "optimized search terms", "rationale": "why this search"},
        {"step": 2, "action": "search", "source": "Federal Register", "query": "...", "rationale": "..."},
        ...
        {"step": N, "action": "synthesize", "description": "combine findings into comprehensive answer"}
    ]
}

Guidelines:
- Tailor search queries to each source's content type
- Include searches for multiple aspects of complex queries
- Always end with a synthesize step
- If the query is broad, break it into sub-topics"""


def create_planner(manager: FoundryAgentManager, model_override: str = None):
    """Create a Planner agent (no tools, planning only)."""
    return manager.create_agent(
        name="planner",
        instructions=PLANNER_INSTRUCTIONS,
        model_override=model_override or manager.fast_model,
    )
