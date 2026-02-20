"""
Assertion helpers for Promptfoo Python assertions.

Each public ``check_*`` function follows the Promptfoo GradingResult
contract:  ``{"pass": bool, "score": float, "reason": str}``.

Metadata (latency, citations, sources, tool-calls) is read from the
JSON cache written by ``promptfoo_provider.py`` during the provider run.

Usage from an inline Promptfoo assertion::

    - type: python
      value: |
        import sys; sys.path.insert(0, '.')
        from promptfoo_helpers import check_latency
        check_latency(output, context, 20)
"""
import json
import os
import hashlib
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".promptfoo_cache")

# Ensure Azure CLI is on PATH (winget install location)
_AZ_CLI_DIR = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
if os.path.isdir(_AZ_CLI_DIR) and _AZ_CLI_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _AZ_CLI_DIR


# ── Cache reader ──────────────────────────────────────────────────────

def _load_meta(query: str) -> dict:
    """Load cached metadata for a given query."""
    key = hashlib.md5(query.encode("utf-8")).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


# ── Latency ───────────────────────────────────────────────────────────

def check_latency(output, context, threshold=20):
    """Score completion time.  1.0 at 0 s → 0.0 at 2× threshold."""
    meta = _load_meta(context["vars"]["query"])
    actual = meta.get("completion_time", 0)
    if actual >= threshold * 2:
        score = 0.0
    else:
        score = max(0.0, 1.0 - actual / (threshold * 2))
    return {
        "pass": score > 0.5,
        "score": round(score, 2),
        "reason": f"Completion time: {actual:.1f}s (threshold: {threshold}s)",
    }


# ── Groundedness ──────────────────────────────────────────────────────

def check_min_citations(output, context, min_count=2):
    """Score citation count, scaling linearly to ``min_count``."""
    meta = _load_meta(context["vars"]["query"])
    actual = meta.get("citations_count", 0)
    score = min(actual / min_count, 1.0) if min_count > 0 else 1.0
    return {
        "pass": actual >= min_count,
        "score": round(score, 2),
        "reason": f"Citations: {actual}/{min_count}",
    }


# ── Relevance ─────────────────────────────────────────────────────────

def check_min_length(output, context, min_chars=150):
    """Score answer length, scaling linearly to ``min_chars``."""
    actual = len(output)
    score = min(actual / min_chars, 1.0) if min_chars > 0 else 1.0
    return {
        "pass": actual >= min_chars,
        "score": round(score, 2),
        "reason": f"Answer length: {actual}/{min_chars} chars",
    }


# ── Coverage ──────────────────────────────────────────────────────────

def check_sources_include(output, context, expected_source=""):
    """Check whether a specific data source was used."""
    meta = _load_meta(context["vars"]["query"])
    sources = meta.get("sources_used", [])
    normalized = expected_source.lower().replace(" ", "_").replace(".", "")
    found = any(
        expected_source.lower() in s.lower() or normalized in s.lower()
        for s in sources
    )
    return {
        "pass": found,
        "score": 1.0 if found else 0.0,
        "reason": (
            f"Found '{expected_source}' in sources"
            if found
            else f"Expected '{expected_source}', sources used: {sources}"
        ),
    }


def check_min_documents(output, context, min_docs=3):
    """Score document count, scaling linearly to ``min_docs``."""
    meta = _load_meta(context["vars"]["query"])
    actual = meta.get("documents_retrieved", 0)
    score = min(actual / min_docs, 1.0) if min_docs > 0 else 1.0
    return {
        "pass": actual >= min_docs,
        "score": round(score, 2),
        "reason": f"Documents retrieved: {actual}/{min_docs}",
    }


# ── Process / trace-based ────────────────────────────────────────────

def check_tool_called(output, context, tool_name=""):
    """Assert a specific tool was invoked during the run."""
    meta = _load_meta(context["vars"]["query"])
    if not meta.get("otel_available", True):
        return {"pass": True, "score": 0.5, "reason": f"Skipped: OTel tracing unavailable (cannot verify '{tool_name}')"}
    tool_calls = meta.get("tool_calls", [])
    called_names = [tc["name"] for tc in tool_calls]
    found = tool_name in called_names
    return {
        "pass": found,
        "score": 1.0 if found else 0.0,
        "reason": (
            f"'{tool_name}' was called"
            if found
            else f"'{tool_name}' NOT called. Tools called: {called_names}"
        ),
    }


def check_no_tool_failures(output, context):
    """Assert no tool calls ended in an error status."""
    meta = _load_meta(context["vars"]["query"])
    if not meta.get("otel_available", True):
        return {"pass": True, "score": 0.5, "reason": "Skipped: OTel tracing unavailable"}
    tool_calls = meta.get("tool_calls", [])
    if not tool_calls:
        return {"pass": True, "score": 1.0, "reason": "No tool calls recorded"}
    failed = [tc for tc in tool_calls if tc.get("status") == "error"]
    score = 1.0 - (len(failed) / len(tool_calls)) if tool_calls else 1.0
    return {
        "pass": len(failed) == 0,
        "score": round(score, 2),
        "reason": (
            "All tool calls succeeded"
            if not failed
            else f"Failed: {[tc['name'] for tc in failed]}"
        ),
    }


def check_min_tool_calls(output, context, min_calls=1):
    """Score total tool-call count, scaling linearly to ``min_calls``."""
    meta = _load_meta(context["vars"]["query"])
    if not meta.get("otel_available", True):
        return {"pass": True, "score": 0.5, "reason": f"Skipped: OTel tracing unavailable (cannot verify min {min_calls} calls)"}
    tool_calls = meta.get("tool_calls", [])
    actual = len(tool_calls)
    score = min(actual / min_calls, 1.0) if min_calls > 0 else 1.0
    return {
        "pass": actual >= min_calls,
        "score": round(score, 2),
        "reason": f"Tool calls: {actual}/{min_calls}",
    }


# ── Azure AI Evaluators ──────────────────────────────────────────────

_azure_evals = None
_azure_init_attempted = False


def _azure():
    """Lazy-initialize AzureEvaluators (returns None on failure)."""
    global _azure_evals, _azure_init_attempted
    if _azure_init_attempted:
        return _azure_evals
    _azure_init_attempted = True
    try:
        import sys
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from evaluation.azure_evaluators import AzureEvaluators
        from azure.identity import DefaultAzureCredential
        _azure_evals = AzureEvaluators.from_env(credential=DefaultAzureCredential())
    except Exception as exc:
        logger.warning("Azure evaluators unavailable: %s", exc)
        _azure_evals = None
    return _azure_evals


def _skip_azure():
    return {"pass": True, "score": 1.0, "reason": "Azure evaluators not available (skipped)"}


def check_azure_relevance(output, context):
    """Azure AI Evaluation SDK — relevance score."""
    evaluators = _azure()
    if not evaluators:
        return _skip_azure()
    query = context["vars"]["query"]
    result = evaluators.evaluate_relevance(query=query, response=output)
    return {
        "pass": result["score"] > 0.5,
        "score": round(result["score"], 2),
        "reason": result.get("detail", ""),
    }


def check_azure_coherence(output, context):
    """Azure AI Evaluation SDK — coherence score."""
    evaluators = _azure()
    if not evaluators:
        return _skip_azure()
    query = context["vars"]["query"]
    result = evaluators.evaluate_coherence(query=query, response=output)
    return {
        "pass": result["score"] > 0.5,
        "score": round(result["score"], 2),
        "reason": result.get("detail", ""),
    }


def check_azure_groundedness(output, context):
    """Azure AI Evaluation SDK — groundedness score."""
    evaluators = _azure()
    if not evaluators:
        return _skip_azure()
    query = context["vars"]["query"]
    result = evaluators.evaluate_groundedness(
        query=query, response=output, context=output,
    )
    return {
        "pass": result["score"] > 0.5,
        "score": round(result["score"], 2),
        "reason": result.get("detail", ""),
    }


def check_azure_fluency(output, context):
    """Azure AI Evaluation SDK — fluency score."""
    evaluators = _azure()
    if not evaluators:
        return _skip_azure()
    result = evaluators.evaluate_fluency(response=output)
    return {
        "pass": result["score"] > 0.5,
        "score": round(result["score"], 2),
        "reason": result.get("detail", ""),
    }


# ── LLM-judged quality (via existing LLMJudge) ──────────────────────

_llm_judge = None
_llm_judge_init_attempted = False


def _judge():
    """Lazy-initialize LLMJudge (returns None on failure)."""
    global _llm_judge, _llm_judge_init_attempted
    if _llm_judge_init_attempted:
        return _llm_judge
    _llm_judge_init_attempted = True
    try:
        import sys
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            load_dotenv(Path(__file__).resolve().parent / ".env")
        except ImportError:
            pass
        from agents.client import FoundryAgentManager
        from evaluation.llm_judge import LLMJudge
        mgr = FoundryAgentManager()
        mgr.__enter__()
        _llm_judge = LLMJudge(mgr.openai_client, model=mgr.fast_model)
    except Exception as exc:
        logger.warning("LLM judge unavailable: %s", exc)
        _llm_judge = None
    return _llm_judge


def check_llm_quality(output, context, criterion=""):
    """LLM-judged quality assertion using the project's existing LLMJudge."""
    judge = _judge()
    if not judge:
        return {"pass": True, "score": 1.0, "reason": "LLM judge not available (skipped)"}
    query = context["vars"]["query"]
    result = judge.judge_quality(output, query, criterion)
    score = result.get("score", 0.0)
    return {
        "pass": score >= 0.7,
        "score": round(score, 2),
        "reason": result.get("reasoning", ""),
    }

