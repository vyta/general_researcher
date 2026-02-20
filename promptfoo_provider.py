"""
Promptfoo custom Python provider for the General Researcher.

Wraps the single_agent architecture and caches structured run metadata
so that assertion helpers can evaluate latency, citations, sources,
tool-call traces, and Azure AI evaluator scores.

Usage (called automatically by Promptfoo):
    promptfoo eval                    # uses promptfooconfig.yaml
    promptfoo eval --no-cache         # skip Promptfoo result cache
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path

# ── Bootstrap imports from src/ ──────────────────────────────────────
_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Ensure Azure CLI is on PATH (winget install location) ────────────
_AZ_CLI_DIR = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
if os.path.isdir(_AZ_CLI_DIR) and _AZ_CLI_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _AZ_CLI_DIR

# ── Cache directory for run metadata ─────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".promptfoo_cache")

# ── Lazy-initialized singletons ──────────────────────────────────────
_manager = None
_architecture = None
_initialized = False


def _init():
    """One-time initialization of tracing, data-sources, and architecture."""
    global _manager, _architecture, _initialized
    if _initialized:
        return

    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(_env_path)
    except ImportError:
        pass

    from utils.tracing import setup_tracing
    from data_sources import get_all_sources
    from agents.client import FoundryAgentManager
    from architectures import ARCHITECTURES

    setup_tracing()

    data_sources = get_all_sources()
    _manager = FoundryAgentManager()
    _manager.__enter__()

    arch_info = ARCHITECTURES["single_agent"]
    _architecture = arch_info["class"](_manager, data_sources)

    os.makedirs(CACHE_DIR, exist_ok=True)
    _initialized = True


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _write_cache(query: str, metadata: dict):
    path = os.path.join(CACHE_DIR, f"{_cache_key(query)}.json")
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2)


# ── Promptfoo entry-point ────────────────────────────────────────────

def call_api(prompt, options, context):
    """
    Called by Promptfoo for every test case.

    Parameters
    ----------
    prompt : str
        The rendered prompt (the research query).
    options : dict
        Provider-level config from the YAML.
    context : dict
        Contains ``vars`` and other test metadata.

    Returns
    -------
    dict
        ``{"output": <answer_text>}`` on success, or
        ``{"error": <message>}`` on failure.
    """
    _init()

    query = prompt.strip()

    # Clear the in-memory span capture before each run
    capture = None
    try:
        from utils.tracing import get_span_capture
        capture = get_span_capture()
        if capture:
            capture.clear()
    except Exception:
        pass

    start = time.time()
    try:
        result = _architecture.research(query, max_results_per_source=5)
        elapsed = time.time() - start
    except Exception as exc:
        elapsed = time.time() - start
        _write_cache(query, {"error": str(exc), "completion_time": round(elapsed, 2)})
        return {"error": str(exc)}

    # ── Gather tool-call spans from the OTel in-memory exporter ──────
    tool_calls = []
    if capture:
        try:
            spans = list(capture.get_finished_spans())
            tool_calls = [
                {
                    "name": s.attributes.get("tool.name", s.name),
                    "status": s.attributes.get("tool.status", "ok"),
                }
                for s in spans
                if s.name.startswith("tool.call.")
            ]
        except Exception:
            pass

    # ── Build metadata and write to cache ────────────────────────────
    citations = result.citations if hasattr(result, "citations") else []
    sources = list(result.sources_checked) if hasattr(result, "sources_checked") else []
    docs = result.documents_retrieved if hasattr(result, "documents_retrieved") else 0

    metadata = {
        "query": query,
        "completion_time": round(elapsed, 2),
        "documents_retrieved": docs,
        "citations_count": len(citations),
        "sources_used": sources,
        "tool_calls": tool_calls,
        "otel_available": capture is not None,
    }

    _write_cache(query, metadata)

    return {"output": result.answer}
