"""
BDD eval runner.

Executes registered scenarios against research architectures,
collects results, and prints scored reports with per-metric breakdowns.
"""
import json
import time
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .dsl import ScenarioBuilder, ScenarioResult, StepResult, get_all_scenarios, METRIC_CATEGORIES
from .steps import match_step
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class ResearchOutput:
    """Normalized output from a research architecture run."""
    query: str
    answer: str
    completion_time: float
    documents_retrieved: int
    citations_count: int
    sources_used: List[str]
    spans: List[Any] = field(default_factory=list)


class EvalRunner:
    """Runs BDD scenarios against architectures and collects scored results."""

    def __init__(self, output_dir: str = "eval_results", llm_judge: Optional[LLMJudge] = None,
                 azure_evaluators=None, verbose: bool = False):
        self.output_dir = output_dir
        self.llm_judge = llm_judge
        self.azure_evaluators = azure_evaluators
        self.verbose = verbose
        import os
        os.makedirs(output_dir, exist_ok=True)

    def run_scenario(
        self,
        scenario: ScenarioBuilder,
        architecture,
        architecture_name: str,
    ) -> ScenarioResult:
        """Run a single scenario against an architecture."""
        if not scenario.query:
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                category=scenario.category,
                architecture=architecture_name,
                error="No query defined in scenario",
            )

        logger.info("Running scenario: %s (%s)", scenario.name, scenario.scenario_id)

        # Clear span capture before running
        from utils.tracing import get_span_capture
        from .actions import match_action

        capture = get_span_capture()
        if capture:
            capture.clear()

        # Execute stages: each when() dispatches an action, then() graders evaluate the result
        context = {"query": scenario.query}
        case_start = time.time()
        current_subject = None
        step_results = []

        for stage_action, stage_thens in scenario.stages:
            # Dispatch the action handler
            handler = match_action(stage_action)
            if handler is None:
                logger.warning("No action handler for: %s", stage_action)
                for assertion, args in stage_thens:
                    step_results.append(StepResult(
                        step_text=f"{assertion} {', '.join(str(a) for a in args)}".strip(),
                        score=0.0,
                        detail=f"No action handler registered for: {stage_action}",
                        stage=stage_action,
                    ))
                continue

            try:
                action_start = time.time()
                raw_result = handler(context, current_subject, architecture)
                action_elapsed = time.time() - action_start
            except Exception as e:
                elapsed = time.time() - case_start
                logger.error("Stage '%s' failed: %s", stage_action, e)
                return ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    scenario_name=scenario.name,
                    category=scenario.category,
                    architecture=architecture_name,
                    completion_time=elapsed,
                    error=f"Stage '{stage_action}' failed: {e}",
                )

            # Normalize output if this action produced a new result
            if raw_result is not current_subject or current_subject is None:
                captured_spans = list(capture.get_finished_spans()) if capture else []
                logger.info("Stage '%s': captured %d spans in %.1fs",
                           stage_action, len(captured_spans), action_elapsed)
                current_subject = _normalize_output(
                    raw_result, scenario.query, action_elapsed, captured_spans
                )

            # Run graders for this stage
            for assertion, args in stage_thens:
                sr = match_step(assertion, args, current_subject,
                               llm_judge=self.llm_judge,
                               azure_evaluators=self.azure_evaluators)
                sr.stage = stage_action
                step_results.append(sr)

        elapsed = time.time() - case_start

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            category=scenario.category,
            architecture=architecture_name,
            steps=step_results,
            completion_time=elapsed,
            documents_retrieved=current_subject.documents_retrieved if current_subject else 0,
            citations_count=current_subject.citations_count if current_subject else 0,
            sources_used=current_subject.sources_used if current_subject else [],
            answer=current_subject.answer if current_subject else "",
        )

    def run_all(
        self,
        architecture,
        architecture_name: str,
        scenarios: Optional[List[ScenarioBuilder]] = None,
    ) -> List[ScenarioResult]:
        """Run all (or specified) scenarios against an architecture."""
        if scenarios is None:
            scenarios = get_all_scenarios()

        results = []
        for i, sc in enumerate(scenarios, 1):
            print(f"\n  [{i}/{len(scenarios)}] {sc.name}")
            print(f"    Query: {sc.query}")

            sr = self.run_scenario(sc, architecture, architecture_name)
            results.append(sr)

            _print_scenario_result(sr, verbose=self.verbose)

        return results

    def print_summary(self, results: List[ScenarioResult], architecture_name: str):
        """Print a scorecard with per-metric breakdowns."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        errored = sum(1 for r in results if r.error)
        successful = [r for r in results if not r.error]

        # Overall score
        overall = sum(r.overall_score for r in successful) / len(successful) if successful else 0.0

        print(f"\n{'=' * 70}")
        print(f"  SCORECARD: {architecture_name}")
        print(f"{'=' * 70}")
        print(f"  Overall: {overall:.2f}  ({passed}/{total} scenarios passed)")
        if errored:
            print(f"  ⚠ {errored} scenario(s) errored")

        # Per-metric scores
        metric_scores: Dict[str, List[float]] = {}
        for r in successful:
            for s in r.steps:
                cat = s.metric or "other"
                metric_scores.setdefault(cat, []).append(s.score)

        if metric_scores:
            print(f"\n  {'Metric':<15} {'Score':>7} {'Steps':>7}")
            print(f"  {'-' * 32}")
            for metric in ["latency", "coverage", "relevance", "groundedness", "quality"]:
                if metric in metric_scores:
                    scores = metric_scores[metric]
                    avg = sum(scores) / len(scores)
                    icon = METRIC_CATEGORIES.get(metric, "")
                    bar = _score_bar(avg)
                    print(f"  {icon} {metric:<13} {avg:>6.2f} {bar}  ({len(scores)})")

        # Per-scenario scores
        print(f"\n  {'Scenario':<30} {'Score':>7} {'Status':>8}")
        print(f"  {'-' * 48}")
        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL" if not r.error else "⚠ ERR"
            name = r.scenario_name[:29]
            print(f"  {name:<30} {r.overall_score:>6.2f}  {status}")

        avg_time = sum(r.completion_time for r in results) / total if total else 0
        print(f"\n  Avg time: {avg_time:.1f}s")
        print(f"{'=' * 70}")

    def save_results(self, results: List[ScenarioResult], architecture_name: str):
        """Save results to JSON with full score data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{architecture_name}_{timestamp}.json"

        successful = [r for r in results if not r.error]
        overall = sum(r.overall_score for r in successful) / len(successful) if successful else 0.0

        # Aggregate metric scores
        metric_scores: Dict[str, List[float]] = {}
        for r in successful:
            for s in r.steps:
                cat = s.metric or "other"
                metric_scores.setdefault(cat, []).append(s.score)

        output = {
            "architecture": architecture_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_score": round(overall, 3),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "total": len(results),
                "metrics": {
                    k: round(sum(v) / len(v), 3) for k, v in metric_scores.items()
                },
            },
            "scenarios": [
                {
                    "id": r.scenario_id,
                    "name": r.scenario_name,
                    "category": r.category,
                    "overall_score": round(r.overall_score, 3),
                    "passed": r.passed,
                    "time": round(r.completion_time, 2),
                    "documents": r.documents_retrieved,
                    "citations": r.citations_count,
                    **({
                        "answer": r.answer,
                    } if self.verbose else {}),
                    "metric_scores": {
                        k: round(v, 3) for k, v in r.scores_by_metric().items()
                    },
                    "error": r.error,
                    "steps": [
                        {
                            "step": s.step_text,
                            "score": round(s.score, 3),
                            "metric": s.metric,
                            "passed": s.passed,
                            "detail": s.detail,
                            "llm_judged": s.is_llm_judged,
                            "stage": s.stage,
                        }
                        for s in r.steps
                    ],
                }
                for r in results
            ],
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  💾 Results saved to: {filename}")


# ── Helpers ───────────────────────────────────────────────────────────

def _count_citations(answer: str) -> int:
    return len(set(re.findall(r"\[(\d+)\]", answer)))


def _normalize_output(raw_result, query: str, elapsed: float, spans: list) -> ResearchOutput:
    """Normalize raw SUT output into a ResearchOutput subject."""
    if isinstance(raw_result, ResearchOutput):
        return raw_result
    if raw_result is None:
        return ResearchOutput(
            query=query, answer="", completion_time=elapsed,
            documents_retrieved=0, citations_count=0, sources_used=[], spans=spans,
        )
    citations_count = (len(raw_result.citations) if hasattr(raw_result, 'citations')
                       else _count_citations(getattr(raw_result, 'answer', '') or ''))
    sources_used = list(raw_result.sources_checked) if hasattr(raw_result, 'sources_checked') else []
    return ResearchOutput(
        query=query,
        answer=getattr(raw_result, 'answer', '') or '',
        completion_time=elapsed,
        documents_retrieved=raw_result.documents_retrieved if hasattr(raw_result, 'documents_retrieved') else 0,
        citations_count=citations_count,
        sources_used=sources_used,
        spans=spans,
    )


def _score_bar(score: float, width: int = 10) -> str:
    """Render a visual score bar like [████████░░]."""
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _print_scenario_result(sr: ScenarioResult, verbose: bool = False):
    """Print a single scenario result with step scores grouped by stage."""
    status = "✓" if sr.passed else "✗"
    print(f"    {status} {sr.scenario_name} — score: {sr.overall_score:.2f} "
          f"({sr.completion_time:.1f}s, {sr.documents_retrieved} docs, {sr.citations_count} cites)")

    if sr.error:
        print(f"      ❌ Error: {sr.error}")
        return

    current_stage = None
    for s in sr.steps:
        if s.stage and s.stage != current_stage:
            current_stage = s.stage
            print(f"      ── {current_stage} ──")
        icon = "✓" if s.passed else "✗"
        suffix = " (LLM)" if s.is_llm_judged else ""
        cat = f"[{s.metric}]" if s.metric else ""
        detail = f" — {s.detail}" if s.detail else ""
        print(f"      {icon} {s.score:.1f}  {cat:<15} {s.step_text}{suffix}{detail}")

    if verbose and sr.answer:
        preview = sr.answer.replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300] + "…"
        print(f"      ── answer ──\n      {preview}")
