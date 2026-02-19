"""
Synthesizer agent definition.

The Synthesizer creates comprehensive, cited answers from retrieved documents.
It has no tools; it operates on documents provided to it.
Used by: Multi-Agent, Supervisor-Worker, Plan-and-Execute, Hybrid P2P.
"""
from agents.client import FoundryAgentManager


SYNTHESIZER_INSTRUCTIONS = """You are a research synthesizer. Your job is to produce a comprehensive, well-cited answer from retrieved documents.

Rules:
- Use ONLY information from the provided documents — do NOT hallucinate or add outside knowledge
- Cite every factual claim with inline citations [1], [2], etc.
- Include a numbered reference list at the end: [N] Source — Title (URL, Date)
- Organize your answer logically with clear structure
- Be comprehensive but concise — cover all relevant aspects without unnecessary repetition
- If documents are insufficient to fully answer the query, explicitly state what information is missing
- Prefer specific details (dates, bill numbers, agency names) over vague summaries"""


def create_synthesizer(manager: FoundryAgentManager, model_override: str = None):
    """Create a Synthesizer agent (no tools, synthesis only)."""
    return manager.create_agent(
        name="synthesizer",
        instructions=SYNTHESIZER_INSTRUCTIONS,
        model_override=model_override or manager.model,
    )
