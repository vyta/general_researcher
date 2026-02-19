"""
BDD-inspired DSL for defining eval scenarios.

Provides a lightweight Given/When/Then framework for expressing
evaluation criteria in a readable, composable way. Each assertion
produces a 0.0–1.0 score; pass/fail is determined by thresholds.

Usage:
    from evaluation.dsl import scenario

    @scenario("AI legislation search", category="legislation")
    def test_ai_legislation(s):
        s.given("a query", "What actions has Congress taken on AI policy?")
        s.when("the agent researches this query")
        s.then("the answer should mention", "artificial intelligence")
        s.then("there should be at least {n} citations", 3)
        s.then("the answer should be", "comprehensive")  # LLM-judged
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any


# ── Metric categories ────────────────────────────────────────────────

METRIC_CATEGORIES = {
    "latency": "⏱️",
    "coverage": "📚",
    "relevance": "🎯",
    "groundedness": "📎",
    "quality": "✨",
}


# ── Result types ──────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of a single Then assertion with a 0.0–1.0 score."""
    step_text: str
    score: float  # 0.0 to 1.0
    metric: str = ""  # metric category (latency, coverage, relevance, etc.)
    detail: str = ""
    is_llm_judged: bool = False

    @property
    def passed(self) -> bool:
        """A step passes if its score is > 0.5 (default threshold)."""
        return self.score > 0.5


@dataclass
class ScenarioResult:
    """Result of running a single scenario against an architecture."""
    scenario_id: str
    scenario_name: str
    category: str
    architecture: str
    steps: List[StepResult] = field(default_factory=list)
    completion_time: float = 0.0
    documents_retrieved: int = 0
    citations_count: int = 0
    sources_used: List[str] = field(default_factory=list)
    answer: str = ""
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Scenario passes if overall score meets threshold (0.7)."""
        return self.overall_score >= 0.7

    @property
    def overall_score(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.score for s in self.steps) / len(self.steps)

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.steps if s.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if not s.passed)

    def scores_by_metric(self) -> dict:
        """Average scores grouped by metric category."""
        groups: dict = {}
        for s in self.steps:
            cat = s.metric or "other"
            groups.setdefault(cat, []).append(s.score)
        return {k: sum(v) / len(v) for k, v in groups.items()}


# ── Scenario builder ─────────────────────────────────────────────────

class ScenarioBuilder:
    """Collects Given/When/Then steps during scenario definition."""

    def __init__(self, scenario_id: str, name: str, category: str):
        self.scenario_id = scenario_id
        self.name = name
        self.category = category
        self.query: Optional[str] = None
        self._thens: List[tuple] = []  # (assertion_text, args)

    def given(self, context: str, value: Any = None):
        """Define a Given step (precondition)."""
        if context == "a query":
            self.query = value
        return self

    def when(self, action: str):
        """Define a When step (action)."""
        return self

    def then(self, assertion: str, *args):
        """Define a Then step (assertion).

        The assertion string is matched against registered step definitions.
        Extra args are passed to the step function.
        """
        self._thens.append((assertion, args))
        return self


# ── Scenario registry ────────────────────────────────────────────────

_SCENARIOS: List[ScenarioBuilder] = []


def scenario(name: str, category: str = "general") -> Callable:
    """Decorator to register a BDD eval scenario.

    The decorated function receives a ScenarioBuilder and should call
    given/when/then to define the scenario's steps.

    The function name (minus 'test_' prefix) becomes the scenario id.
    """
    def decorator(fn: Callable) -> Callable:
        sid = fn.__name__
        if sid.startswith("test_"):
            sid = sid[5:]
        builder = ScenarioBuilder(scenario_id=sid, name=name, category=category)
        fn(builder)
        _SCENARIOS.append(builder)
        return fn
    return decorator


def get_all_scenarios() -> List[ScenarioBuilder]:
    """Return all registered scenarios."""
    return list(_SCENARIOS)


def get_scenarios_by_category(category: str) -> List[ScenarioBuilder]:
    """Return scenarios matching a category."""
    return [s for s in _SCENARIOS if s.category == category]
