"""
Step definitions for BDD eval assertions.

Each step produces a 0.0–1.0 score and a metric category.
Steps are matched by assertion string prefix.

Metric categories:
  - latency: response time measurements
  - coverage: source diversity, document retrieval breadth
  - relevance: topic matching, keyword presence
  - groundedness: citation quality, evidence backing
  - quality: LLM-judged answer quality (comprehensive, well-structured, etc.)
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, TYPE_CHECKING

from .dsl import StepResult

if TYPE_CHECKING:
    from .runner import ResearchOutput


# ── Step registry ─────────────────────────────────────────────────────

_STEP_DEFS: list = []  # [(pattern, fn, is_llm)]


def step(pattern: str, is_llm: bool = False):
    """Register a step definition."""
    def decorator(fn):
        _STEP_DEFS.append((pattern, fn, is_llm))
        return fn
    return decorator


def match_step(assertion: str, args: tuple, output: "ResearchOutput", llm_judge=None) -> StepResult:
    """Find and execute a matching step definition."""
    for pattern, fn, is_llm in _STEP_DEFS:
        if assertion.lower().startswith(pattern.lower()):
            if is_llm and llm_judge is None:
                return StepResult(
                    step_text=_format_step(assertion, args),
                    score=1.0,
                    metric="quality",
                    detail="Skipped (no LLM judge configured)",
                    is_llm_judged=True,
                )
            return fn(assertion, args, output, llm_judge)

    return StepResult(
        step_text=_format_step(assertion, args),
        score=0.0,
        detail=f"No step definition found for: {assertion}",
    )


def _format_step(assertion: str, args: tuple) -> str:
    if args:
        return f"{assertion} {', '.join(str(a) for a in args)}"
    return assertion


# ── Relevance steps (keyword/topic matching) ──────────────────────────

@step("the answer should mention")
def _answer_should_mention(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    phrase = str(args[0]) if args else ""
    found = phrase.lower() in output.answer.lower()
    return StepResult(
        step_text=f"the answer should mention \"{phrase}\"",
        score=1.0 if found else 0.0,
        metric="relevance",
        detail="" if found else f"'{phrase}' not found in answer",
    )


@step("the answer should not mention")
def _answer_should_not_mention(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    phrase = str(args[0]) if args else ""
    found = phrase.lower() in output.answer.lower()
    return StepResult(
        step_text=f"the answer should not mention \"{phrase}\"",
        score=0.0 if found else 1.0,
        metric="relevance",
        detail="" if not found else f"'{phrase}' was found in answer",
    )


# ── Groundedness steps (citations, evidence) ─────────────────────────

@step("there should be at least")
def _min_citations(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    n = int(args[0]) if args else _extract_number(assertion)
    what = "citations" if "citation" in assertion.lower() else "items"
    actual = output.citations_count
    # Score scales linearly: 0 at 0, 1.0 at n, capped at 1.0
    score = min(actual / n, 1.0) if n > 0 else 1.0
    return StepResult(
        step_text=f"there should be at least {n} {what}",
        score=score,
        metric="groundedness",
        detail=f"{actual}/{n}",
    )


@step("the answer should be at least")
def _min_length(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    n = int(args[0]) if args else _extract_number(assertion)
    actual = len(output.answer)
    score = min(actual / n, 1.0) if n > 0 else 1.0
    return StepResult(
        step_text=f"the answer should be at least {n} characters",
        score=score,
        metric="relevance",
        detail=f"{actual}/{n} chars",
    )


# ── Coverage steps (source diversity, documents) ─────────────────────

@step("sources should include")
def _sources_include(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    source = str(args[0]) if args else ""
    source_lower = source.lower().replace(" ", "_").replace(".", "")
    found = any(
        source.lower() in s.lower() or source_lower in s.lower()
        for s in output.sources_used
    )
    return StepResult(
        step_text=f"sources should include \"{source}\"",
        score=1.0 if found else 0.0,
        metric="coverage",
        detail="" if found else f"sources used: {output.sources_used}",
    )


@step("documents retrieved should be at least")
def _min_documents(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    n = int(args[0]) if args else _extract_number(assertion)
    actual = output.documents_retrieved
    score = min(actual / n, 1.0) if n > 0 else 1.0
    return StepResult(
        step_text=f"documents retrieved should be at least {n}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{n}",
    )


# ── Latency steps ────────────────────────────────────────────────────

@step("completion time should be under")
def _max_time(assertion: str, args: tuple, output: "ResearchOutput", _) -> StepResult:
    threshold = float(args[0]) if args else _extract_number(assertion)
    actual = output.completion_time
    # Score scales linearly: 1.0 at 0s, 0.5 at threshold, 0.0 at 2x threshold
    if actual >= threshold * 2:
        score = 0.0
    else:
        score = max(0.0, 1.0 - actual / (threshold * 2))
    return StepResult(
        step_text=f"completion time should be under {threshold}s",
        score=round(score, 2),
        metric="latency",
        detail=f"{actual:.1f}s (threshold: {threshold}s)",
    )


# ── Quality steps (LLM-judged) ───────────────────────────────────────

@step("the answer should be", is_llm=True)
def _answer_quality(assertion: str, args: tuple, output: "ResearchOutput", llm_judge) -> StepResult:
    quality = str(args[0]) if args else ""
    result = llm_judge.judge_quality(output.answer, output.query, quality)
    return StepResult(
        step_text=f"the answer should be \"{quality}\"",
        score=result.get("score", 1.0 if result.get("passed") else 0.0),
        metric="quality",
        detail=result.get("reasoning", ""),
        is_llm_judged=True,
    )


@step("the answer should", is_llm=True)
def _answer_should(assertion: str, args: tuple, output: "ResearchOutput", llm_judge) -> StepResult:
    criteria = str(args[0]) if args else assertion.replace("the answer should ", "")
    result = llm_judge.judge_criteria(output.answer, output.query, criteria)
    return StepResult(
        step_text=f"the answer should {criteria}",
        score=result.get("score", 1.0 if result.get("passed") else 0.0),
        metric="quality",
        detail=result.get("reasoning", ""),
        is_llm_judged=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_number(text: str) -> int:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0
