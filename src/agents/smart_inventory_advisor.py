"""
Smart Inventory Advisor — ACP agent definition.

This agent is a GitHub Copilot custom agent ("Material Substitution")
that investigates material substitutions in an industrial inventory
knowledge graph. Unlike the Foundry agents (researcher, critic, etc.),
it runs via Copilot CLI and communicates over the Agent Client Protocol
(ACP).

The agent definition lives in the parent repo:
    .github/agents/material-substitution.agent.md

It is NOT registered via Foundry and does NOT use FoundryAgentManager.
Instead, it is spawned (or connected to) via ACP transport and tested
through the ACPAgentOrchestrator architecture.

Capabilities (owned by the agent, opaque to this framework):
    - RDF Explorer MCP — SPARQL queries against a knowledge graph
    - Databricks MCP   — SQL queries for operational/financial data
    - PDF Reader MCP   — technical specification extraction
    - VS Code tools    — file read/write/search
"""
from dataclasses import dataclass, field
from typing import List, Optional


# Path to the agent definition relative to the repository root
AGENT_DEFINITION_PATH = ".github/agents/material-substitution.agent.md"


@dataclass
class ACPAgentConfig:
    """Connection configuration for an ACP agent.

    Supports two transport modes:
        stdio — spawn the agent as a subprocess (recommended)
        tcp   — connect to an already-running ACP server
    """

    name: str
    """Human-readable name shown in eval reports."""

    transport: str = "stdio"
    """Transport mode: 'stdio' or 'tcp'."""

    # stdio transport settings
    executable: str = "copilot"
    """Path or command name for the Copilot CLI binary."""

    extra_args: List[str] = field(default_factory=list)
    """Additional CLI arguments passed to the executable."""

    cwd: Optional[str] = None
    """Working directory for the subprocess. When None the repo root
    is used so the agent can discover its .agent.md definition."""

    # tcp transport settings
    host: str = "localhost"
    """Hostname for TCP transport."""

    port: int = 3000
    """Port for TCP transport."""

    # shared
    timeout: int = 120
    """Maximum seconds to wait for a prompt response."""


# ── Default configuration ────────────────────────────────────────────

SMART_INVENTORY_ADVISOR = ACPAgentConfig(
    name="Smart Inventory Advisor",
    transport="stdio",
    executable="copilot",
    extra_args=["--agent", "material-substitution", "--allow-all"],
    # cwd defaults to repo root so Copilot CLI can find
    # .github/agents/material-substitution.agent.md
)

