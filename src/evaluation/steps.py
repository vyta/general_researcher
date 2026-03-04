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

import logging
import re
from typing import Any, List, Optional, TYPE_CHECKING

from .dsl import StepResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .runner import ResearchOutput


# ── Step registry ─────────────────────────────────────────────────────

_STEP_DEFS: list = []  # [(pattern, fn, is_llm, is_azure_eval)]


def step(pattern: str, is_llm: bool = False, is_azure_eval: bool = False):
    """Register a step definition."""
    def decorator(fn):
        _STEP_DEFS.append((pattern, fn, is_llm, is_azure_eval))
        return fn
    return decorator


def match_step(assertion: str, args: tuple, output: "ResearchOutput",
               llm_judge=None, azure_evaluators=None) -> StepResult:
    """
    Find and execute a matching step definition (longest pattern match).
    
    Uses longest-pattern-first dispatch to resolve overlapping patterns deterministically.
    For example, "the answer should mention one of" will match the specific
    @step("the answer should mention one of") instead of the generic
    @step("the answer should mention"), regardless of registration order.
    
    Args:
        assertion: The assertion text to match (e.g., "the answer should mention AI")
        args: Additional arguments for the assertion
        output: Research output to validate against
        llm_judge: Optional LLM judge for quality assessments
        azure_evaluators: Optional Azure AI evaluators
    
    Returns:
        StepResult with score, metric, and details
    """
    # Find all matching patterns
    matches = [
        (pattern, fn, is_llm, is_azure_eval)
        for pattern, fn, is_llm, is_azure_eval in _STEP_DEFS
        if assertion.lower().startswith(pattern.lower())
    ]
    
    if matches:
        # Use longest pattern (most specific match) - ensures order-independent dispatch
        pattern, fn, is_llm, is_azure_eval = max(matches, key=lambda m: len(m[0]))
        
        if is_llm and llm_judge is None:
            return StepResult(
                step_text=_format_step(assertion, args),
                score=1.0,
                metric="quality",
                detail="Skipped (no LLM judge configured)",
                is_llm_judged=True,
            )
        if is_azure_eval and azure_evaluators is None:
            return StepResult(
                step_text=_format_step(assertion, args),
                score=1.0,
                metric=_infer_metric(pattern),
                detail="Skipped (Azure evaluators not configured)",
                is_llm_judged=True,
            )
        return fn(assertion, args, output, llm_judge, azure_evaluators)

    return StepResult(
        step_text=_format_step(assertion, args),
        score=0.0,
        detail=f"No step definition found for: {assertion}",
    )


def _format_step(assertion: str, args: tuple) -> str:
    if args:
        return f"{assertion} {', '.join(str(a) for a in args)}"
    return assertion


def _infer_metric(pattern: str) -> str:
    """Infer metric category from step pattern for skip messages."""
    if "relevance" in pattern or "relevant" in pattern:
        return "relevance"
    if "coherence" in pattern or "coherent" in pattern:
        return "quality"
    if "groundedness" in pattern or "grounded" in pattern:
        return "groundedness"
    if "fluency" in pattern or "fluent" in pattern:
        return "quality"
    return "quality"


# ── Relevance steps (keyword/topic matching) ──────────────────────────

@step("the answer should mention one of")
def _answer_should_mention_one_of(assertion, args, output, _, __) -> StepResult:
    terms = args[0] if args and isinstance(args[0], list) else list(args)
    answer_lower = output.answer.lower()
    matched = [t for t in terms if t.lower() in answer_lower]
    found = len(matched) > 0
    return StepResult(
        step_text=f"the answer should mention one of {terms}",
        score=1.0 if found else 0.0,
        metric="relevance",
        detail=f"matched: {matched}" if found else f"none of {terms} found in answer",
    )


@step("the answer should mention")
def _answer_should_mention(assertion, args, output, _, __) -> StepResult:
    phrase = str(args[0]) if args else ""
    found = phrase.lower() in output.answer.lower()
    return StepResult(
        step_text=f"the answer should mention \"{phrase}\"",
        score=1.0 if found else 0.0,
        metric="relevance",
        detail="" if found else f"'{phrase}' not found in answer",
    )


@step("the answer should not mention")
def _answer_should_not_mention(assertion, args, output, _, __) -> StepResult:
    phrase = str(args[0]) if args else ""
    found = phrase.lower() in output.answer.lower()
    return StepResult(
        step_text=f"the answer should not mention \"{phrase}\"",
        score=0.0 if found else 1.0,
        metric="relevance",
        detail="" if not found else f"'{phrase}' was found in answer",
    )


@step("the answer should contain a number")
def _answer_contains_number(assertion, args, output, _, __) -> StepResult:
    import re as _re
    numbers = _re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', output.answer)
    found = len(numbers) > 0
    return StepResult(
        step_text="the answer should contain a number",
        score=1.0 if found else 0.0,
        metric="relevance",
        detail=f"found {len(numbers)} numbers" if found else "no numeric values found",
    )


# ── Groundedness steps (citations, evidence) ─────────────────────────

@step("there should be at least")
def _min_citations(assertion, args, output, _, __) -> StepResult:
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
def _min_length(assertion, args, output, _, __) -> StepResult:
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
def _sources_include(assertion, args, output, _, __) -> StepResult:
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
def _min_documents(assertion, args, output, _, __) -> StepResult:
    n = int(args[0]) if args else _extract_number(assertion)
    actual = output.documents_retrieved
    score = min(actual / n, 1.0) if n > 0 else 1.0
    return StepResult(
        step_text=f"documents retrieved should be at least {n}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{n}",
    )


@step("unique sources used should be at least")
def _min_unique_sources(assertion, args, output, _, __) -> StepResult:
    n = int(args[0]) if args else _extract_number(assertion)
    actual = len(output.sources_used)
    score = min(actual / n, 1.0) if n > 0 else 1.0
    return StepResult(
        step_text=f"unique sources used should be at least {n}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{n} sources: {output.sources_used}",
    )


# ── Latency steps ────────────────────────────────────────────────────

@step("completion time should be under")
def _max_time(assertion, args, output, _, __) -> StepResult:
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
def _answer_quality(assertion, args, output, llm_judge, _) -> StepResult:
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
def _answer_should(assertion, args, output, llm_judge, _) -> StepResult:
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


# ── Azure AI Evaluation steps ────────────────────────────────────────

@step("azure relevance score", is_azure_eval=True)
def _azure_relevance(assertion, args, output, _, azure_evaluators) -> StepResult:
    result = azure_evaluators.evaluate_relevance(query=output.query, response=output.answer)
    return StepResult(
        step_text="azure relevance score",
        score=result["score"],
        metric="relevance",
        detail=result["detail"],
        is_llm_judged=True,
    )


@step("azure coherence score", is_azure_eval=True)
def _azure_coherence(assertion, args, output, _, azure_evaluators) -> StepResult:
    result = azure_evaluators.evaluate_coherence(query=output.query, response=output.answer)
    return StepResult(
        step_text="azure coherence score",
        score=result["score"],
        metric="quality",
        detail=result["detail"],
        is_llm_judged=True,
    )


@step("azure groundedness score", is_azure_eval=True)
def _azure_groundedness(assertion, args, output, _, azure_evaluators) -> StepResult:
    # Use the answer itself as context (citations inline)
    context = args[0] if args else output.answer
    result = azure_evaluators.evaluate_groundedness(
        query=output.query, response=output.answer, context=context
    )
    return StepResult(
        step_text="azure groundedness score",
        score=result["score"],
        metric="groundedness",
        detail=result["detail"],
        is_llm_judged=True,
    )


@step("azure fluency score", is_azure_eval=True)
def _azure_fluency(assertion, args, output, _, azure_evaluators) -> StepResult:
    result = azure_evaluators.evaluate_fluency(response=output.answer)
    return StepResult(
        step_text="azure fluency score",
        score=result["score"],
        metric="quality",
        detail=result["detail"],
        is_llm_judged=True,
    )


# ── Process steps (trace-based) ──────────────────────────────────────

def _get_tool_spans(output) -> list:
    """Extract tool call spans from captured OTel spans."""
    return [s for s in getattr(output, 'spans', []) if s.name.startswith("tool.call.")]


def _get_agent_spans(output) -> list:
    """Extract agent run spans from captured OTel spans."""
    return [s for s in getattr(output, 'spans', []) if s.name.startswith("agent.run.")]


def _get_named_agent_spans(output, agent_name: str) -> list:
    """Extract spans for a specific agent by name."""
    return [s for s in _get_agent_spans(output)
            if s.attributes.get("agent.name", "") == agent_name]


@step("the agent should have called")
def _agent_called_tool(assertion, args, output, _, __) -> StepResult:
    tool_name = str(args[0]) if args else ""
    tool_spans = _get_tool_spans(output)
    called_tools = [s.attributes.get("tool.name", "") for s in tool_spans]
    found = tool_name in called_tools
    return StepResult(
        step_text=f"the agent should have called \"{tool_name}\"",
        score=1.0 if found else 0.0,
        metric="coverage",
        detail="" if found else f"tools called: {called_tools}",
    )


@step("total tool calls should be at most")
def _max_tool_calls(assertion, args, output, _, __) -> StepResult:
    max_calls = int(args[0]) if args else _extract_number(assertion)
    tool_spans = _get_tool_spans(output)
    actual = len(tool_spans)
    score = min(max_calls / actual, 1.0) if actual > 0 else 1.0
    return StepResult(
        step_text=f"total tool calls should be at most {max_calls}",
        score=score,
        metric="latency",
        detail=f"{actual}/{max_calls}",
    )


@step("total tool calls should be at least")
def _min_tool_calls(assertion, args, output, _, __) -> StepResult:
    min_calls = int(args[0]) if args else _extract_number(assertion)
    tool_spans = _get_tool_spans(output)
    actual = len(tool_spans)
    score = min(actual / min_calls, 1.0) if min_calls > 0 else 1.0
    return StepResult(
        step_text=f"total tool calls should be at least {min_calls}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{min_calls}",
    )


@step("no tool calls should have failed")
def _no_tool_failures(assertion, args, output, _, __) -> StepResult:
    tool_spans = _get_tool_spans(output)
    if not tool_spans:
        return StepResult(
            step_text="no tool calls should have failed",
            score=1.0,
            metric="quality",
            detail="no tool calls recorded",
        )
    failed = [s for s in tool_spans if s.attributes.get("tool.status") == "error"]
    total = len(tool_spans)
    score = 1.0 - (len(failed) / total) if total > 0 else 1.0
    failed_names = [s.attributes.get("tool.name", "?") for s in failed]
    return StepResult(
        step_text="no tool calls should have failed",
        score=score,
        metric="quality",
        detail="" if not failed else f"failed: {failed_names}",
    )


@step("agent runs should be at most")
def _max_agent_runs(assertion, args, output, _, __) -> StepResult:
    max_runs = int(args[0]) if args else _extract_number(assertion)
    agent_spans = _get_agent_spans(output)
    actual = len(agent_spans)
    score = min(max_runs / actual, 1.0) if actual > 0 else 1.0
    return StepResult(
        step_text=f"agent runs should be at most {max_runs}",
        score=score,
        metric="latency",
        detail=f"{actual}/{max_runs}",
    )


@step("no redundant tool calls")
def _no_redundant_tool_calls(assertion, args, output, _, __) -> StepResult:
    tool_spans = _get_tool_spans(output)
    seen = set()
    redundant = []
    for s in tool_spans:
        key = (s.attributes.get("tool.name", ""), s.attributes.get("tool.arguments", ""))
        if key in seen:
            redundant.append(s.attributes.get("tool.name", "?"))
        seen.add(key)
    total = len(tool_spans)
    score = 1.0 - (len(redundant) / total) if total > 0 else 1.0
    return StepResult(
        step_text="no redundant tool calls",
        score=score,
        metric="latency",
        detail="" if not redundant else f"{len(redundant)} redundant: {redundant}",
    )


@step("search queries should be at least")
def _min_search_queries(assertion, args, output, _, __) -> StepResult:
    """Count distinct (tool, query) pairs across search tool calls."""
    import json as _json
    min_queries = int(args[0]) if args else _extract_number(assertion)
    tool_spans = _get_tool_spans(output)
    queries = set()
    for s in tool_spans:
        name = s.attributes.get("tool.name", "")
        if not name.startswith("search_"):
            continue
        raw_args = s.attributes.get("tool.arguments", "")
        try:
            parsed = _json.loads(raw_args)
            q = parsed.get("query", "")
        except (ValueError, AttributeError):
            q = raw_args
        queries.add((name, q))
    actual = len(queries)
    score = min(actual / min_queries, 1.0) if min_queries > 0 else 1.0
    return StepResult(
        step_text=f"search queries should be at least {min_queries}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{min_queries} distinct queries",
    )


# ── Architecture-specific process steps ──────────────────────────────

@step("code should have been executed")
def _code_executed(assertion, args, output, _, __) -> StepResult:
    tool_spans = _get_tool_spans(output)
    code_spans = [s for s in tool_spans if s.attributes.get("tool.name") == "execute_python"]
    found = len(code_spans) > 0
    return StepResult(
        step_text="code should have been executed",
        score=1.0 if found else 0.0,
        metric="coverage",
        detail=f"{len(code_spans)} code executions" if found else "execute_python never called",
    )


@step("no code execution errors")
def _no_code_errors(assertion, args, output, _, __) -> StepResult:
    tool_spans = _get_tool_spans(output)
    code_spans = [s for s in tool_spans if s.attributes.get("tool.name") == "execute_python"]
    if not code_spans:
        return StepResult(
            step_text="no code execution errors",
            score=1.0,
            metric="quality",
            detail="no code executions recorded",
        )
    failed = [s for s in code_spans if s.attributes.get("tool.status") == "error"]
    score = 1.0 - (len(failed) / len(code_spans)) if code_spans else 1.0
    return StepResult(
        step_text="no code execution errors",
        score=score,
        metric="quality",
        detail="" if not failed else f"{len(failed)}/{len(code_spans)} executions failed",
    )


@step("the critic should have run")
def _critic_ran(assertion, args, output, _, __) -> StepResult:
    critic_spans = _get_named_agent_spans(output, "critic")
    found = len(critic_spans) > 0
    return StepResult(
        step_text="the critic should have run",
        score=1.0 if found else 0.0,
        metric="coverage",
        detail=f"{len(critic_spans)} critic run(s)" if found else "critic never ran",
    )


@step("critic iterations should be at most")
def _max_critic_iterations(assertion, args, output, _, __) -> StepResult:
    max_iters = int(args[0]) if args else _extract_number(assertion)
    critic_spans = _get_named_agent_spans(output, "critic")
    actual = len(critic_spans)
    score = min(max_iters / actual, 1.0) if actual > 0 else 1.0
    return StepResult(
        step_text=f"critic iterations should be at most {max_iters}",
        score=score,
        metric="latency",
        detail=f"{actual}/{max_iters} iterations",
    )


@step("the planner should have run")
def _planner_ran(assertion, args, output, _, __) -> StepResult:
    planner_spans = _get_named_agent_spans(output, "planner")
    found = len(planner_spans) > 0
    return StepResult(
        step_text="the planner should have run",
        score=1.0 if found else 0.0,
        metric="coverage",
        detail=f"{len(planner_spans)} planner run(s)" if found else "planner never ran",
    )


@step("the synthesizer should have run")
def _synthesizer_ran(assertion, args, output, _, __) -> StepResult:
    synth_spans = _get_named_agent_spans(output, "synthesizer")
    found = len(synth_spans) > 0
    return StepResult(
        step_text="the synthesizer should have run",
        score=1.0 if found else 0.0,
        metric="coverage",
        detail=f"{len(synth_spans)} synthesizer run(s)" if found else "synthesizer never ran",
    )


@step("source workers should have run at least")
def _min_source_workers(assertion, args, output, _, __) -> StepResult:
    min_workers = int(args[0]) if args else _extract_number(assertion)
    agent_spans = _get_agent_spans(output)
    # Source workers have names like "worker_govinfo", "worker_federal_register", etc.
    worker_spans = [s for s in agent_spans
                    if "worker" in s.attributes.get("agent.name", "").lower()]
    actual = len(worker_spans)
    score = min(actual / min_workers, 1.0) if min_workers > 0 else 1.0
    return StepResult(
        step_text=f"source workers should have run at least {min_workers}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{min_workers} workers",
    )


@step("distinct agents should have run at least")
def _min_distinct_agents(assertion, args, output, _, __) -> StepResult:
    min_agents = int(args[0]) if args else _extract_number(assertion)
    agent_spans = _get_agent_spans(output)
    distinct = set(s.attributes.get("agent.name", "") for s in agent_spans)
    actual = len(distinct)
    score = min(actual / min_agents, 1.0) if min_agents > 0 else 1.0
    return StepResult(
        step_text=f"distinct agents should have run at least {min_agents}",
        score=score,
        metric="coverage",
        detail=f"{actual}/{min_agents} agents: {sorted(distinct)}",
    )


# ── Pattern validation ───────────────────────────────────────────────

def _validate_patterns():
    """
    Validate step patterns at import time.
    
    Checks for overlapping patterns that could cause matching ambiguities.
    Since we use longest-match-first dispatch, overlaps are handled correctly,
    but this validation helps developers understand the pattern hierarchy.
    
    Warnings are issued for informational purposes only and do not block execution.
    """
    patterns = [p for p, _, _, _ in _STEP_DEFS]
    warnings = []
    
    for i, pattern1 in enumerate(patterns):
        for j, pattern2 in enumerate(patterns):
            if i != j and pattern1.lower() != pattern2.lower():
                # Check if pattern1 is substring of pattern2
                if pattern1.lower() in pattern2.lower():
                    warnings.append(
                        f"Overlapping patterns detected: '{pattern1}' is substring of '{pattern2}'. "
                        f"Longest-match-first dispatch is active (longer pattern takes precedence)."
                    )
    
    if warnings:
        # Deduplicate warnings (each overlap found twice due to pairwise check)
        seen = set()
        for warning in warnings:
            if warning not in seen:
                logger.debug(warning)
                seen.add(warning)


# Call validation at module import time
_validate_patterns()
