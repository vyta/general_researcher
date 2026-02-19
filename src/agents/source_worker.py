"""
Source Worker agent definition.

A Source Worker is specialized to search a single data source and summarize findings.
It is parameterized at creation time with the source name and its specific search tool.
Used by: Supervisor-Worker (×3), Hybrid P2P (×3).
"""
from agents.client import FoundryAgentManager
from data_sources.base import DataSource
from tools.search_tools import get_source_search_tool


SOURCE_DESCRIPTIONS = {
    "GovInfo": "Congressional bills, legislation text, committee reports, and legislative actions",
    "Federal Register": "Federal regulations, proposed rules, executive orders, and agency notices",
    "Data.gov": "Government datasets, data catalogs, statistical data, and program information",
}

SOURCE_WORKER_INSTRUCTIONS = """You are a specialized research worker for {source_name}.

Source description: {source_description}

When given a research query:
1. Search your assigned source with optimized search terms appropriate for this source type
2. Review results and refine your search if initial results are insufficient
3. Summarize your key findings concisely

Respond with JSON only:
{{
    "documents_found": <number>,
    "summary": "2-3 sentence summary of what you found",
    "key_discoveries": ["specific finding 1", "specific finding 2", ...],
    "search_queries_used": ["query1", "query2"]
}}"""


def create_source_worker(
    manager: FoundryAgentManager,
    source: DataSource,
    model_override: str = None,
):
    """Create a Source Worker agent for one specific data source.
    
    Returns (agent, tools, stats) tuple.
    """
    description = SOURCE_DESCRIPTIONS.get(source.name, "Government information")
    tools, stats = get_source_search_tool(source)

    agent = manager.create_agent(
        name=f"{source.name.lower().replace(' ', '-').replace('.', '')}-worker",
        instructions=SOURCE_WORKER_INSTRUCTIONS.format(
            source_name=source.name,
            source_description=description,
        ),
        tools=tools.definitions,
        model_override=model_override or manager.fast_model,
    )
    return agent, tools, stats
