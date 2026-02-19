"""Shared data types and helpers for all architectures."""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ResearchResult:
    """Common result container for all architecture orchestrations.

    The ``metadata`` dict carries architecture-specific details (e.g.
    number of rounds, critic scores, plan steps).
    """
    query: str
    answer: str
    sources_checked: List[str]
    documents_retrieved: int
    documents_used: int
    citations: List[Dict[str, str]]
    time_elapsed: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_citations(answer: str) -> List[Dict[str, str]]:
    """Extract citation references [1], [2], ... from an answer string.

    Returns a list of dicts with ``number`` as a string key. Detailed
    source metadata is not available from the raw text — architectures
    can enrich these later if document lists are available.
    """
    numbers = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer)))
    return [{"number": str(n)} for n in numbers]

