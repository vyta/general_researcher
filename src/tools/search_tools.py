"""
FunctionTool wrappers for data sources.

Bridges the existing DataSource.search() interface to Foundry Agent tools.
Each source is wrapped as a callable function returning JSON for FunctionTool.
"""
import json
import logging
import threading
from typing import List

from azure.ai.agents.models import FunctionTool

from data_sources.base import DataSource, RetrievedDocument
from data_sources import get_all_sources

logger = logging.getLogger(__name__)


class ToolCallStats:
    """Thread-safe counter for tool call statistics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.documents_retrieved = 0
        self.sources_called: List[str] = []

    def record(self, fn_name: str, output: str):
        with self._lock:
            if fn_name not in self.sources_called:
                self.sources_called.append(fn_name)
            try:
                parsed = json.loads(output)
                if isinstance(parsed, list):
                    self.documents_retrieved += len(parsed)
                elif isinstance(parsed, dict) and "error" not in parsed:
                    self.documents_retrieved += 1
            except (json.JSONDecodeError, TypeError):
                pass

    def reset(self):
        with self._lock:
            docs = self.documents_retrieved
            sources = list(self.sources_called)
            self.documents_retrieved = 0
            self.sources_called.clear()
            return docs, sources


def _serialize_documents(documents: List[RetrievedDocument]) -> str:
    """Serialize retrieved documents to JSON for agent consumption."""
    return json.dumps([
        {
            "title": doc.title,
            "source": doc.source,
            "url": doc.url,
            "date": doc.date.isoformat() if doc.date else "",
            "content": doc.content[:1000] if doc.content else "",
        }
        for doc in documents
    ], indent=2)


def _make_search_fn(source: DataSource, stats: ToolCallStats):
    """Create a search function for a specific data source."""
    safe_name = source.name.lower().replace(" ", "_").replace(".", "")

    def search(query: str, max_results: int = 5) -> str:
        logger.info("search_%s called: query=%r max_results=%d", safe_name, query, max_results)
        result = source.search(query, max_results)
        if result.error:
            logger.warning("search_%s error: %s", safe_name, result.error)
            output = json.dumps({"error": result.error, "documents": []})
            stats.record(f"search_{safe_name}", output)
            return output
        logger.info("search_%s returned %d documents", safe_name, len(result.documents))
        output = _serialize_documents(result.documents)
        stats.record(f"search_{safe_name}", output)
        return output

    search.__name__ = f"search_{safe_name}"
    search.__doc__ = f"Search {source.name} for government documents matching the query. Returns JSON array of results with title, source, url, date, and content."
    search.__annotations__ = {"query": str, "max_results": int, "return": str}

    return search


def get_all_search_tools(govinfo_api_key: str = None) -> tuple:
    """Create a FunctionTool containing search functions for ALL data sources.

    Returns (FunctionTool, ToolCallStats) — stats can be reset between runs
    to get per-run document counts.
    """
    stats = ToolCallStats()
    sources = get_all_sources(govinfo_api_key=govinfo_api_key)
    functions = {_make_search_fn(source, stats) for source in sources}
    tool = FunctionTool(functions=functions)
    fn_names = [f.__name__ for f in functions]
    logger.info("Created FunctionTool with %d search functions: %s", len(functions), fn_names)
    logger.debug("Tool definitions: %s", tool.definitions)
    return tool, stats


def get_source_search_tool(source: DataSource) -> tuple:
    """Create a FunctionTool for a single data source.

    Returns (FunctionTool, ToolCallStats).
    """
    stats = ToolCallStats()
    fn = _make_search_fn(source, stats)
    tool = FunctionTool(functions={fn})
    logger.info("Created FunctionTool for source %s: %s", source.name, fn.__name__)
    return tool, stats
