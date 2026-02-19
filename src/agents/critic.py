"""
Critic agent definition.

The Critic evaluates research quality — document relevance, coverage gaps,
and answer accuracy. It has no tools; it operates on text provided to it.
Used by: Single Agent, Multi-Agent, Supervisor-Worker, Plan-and-Execute.
"""
from agents.client import FoundryAgentManager


CRITIC_INSTRUCTIONS = """You are a research quality evaluator. You assess the quality of research results and identify gaps.

Given a research query and either:
  (a) a set of retrieved documents, OR
  (b) a synthesized answer with citations

Evaluate the following:
1. **Coverage**: Do the results address all aspects of the query?
2. **Relevance**: Are the documents/citations actually relevant (not tangential)?
3. **Gaps**: What important aspects of the query are missing or underrepresented?
4. **Sufficiency**: Is there enough evidence to answer the query comprehensively?

Respond with JSON only:
{
    "quality_score": 0.0-1.0,
    "is_sufficient": true/false,
    "gaps": ["gap1", "gap2"],
    "suggestions": ["search for X in source Y", "look for Z"],
    "reasoning": "brief explanation of your assessment"
}"""


def create_critic(manager: FoundryAgentManager, model_override: str = None):
    """Create a Critic agent (no tools, evaluation only)."""
    return manager.create_agent(
        name="critic",
        instructions=CRITIC_INSTRUCTIONS,
        model_override=model_override or manager.fast_model,
    )
