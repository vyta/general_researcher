"""
Foundry Agent lifecycle manager (v2 API).

Uses the Foundry Agents v2 REST API for agent registration (portal visibility)
and the OpenAI Responses API for invocation.

Agent lifecycle:
  1. Register agent via REST: POST .../agents/{name}/versions
  2. Invoke via OpenAI Responses API with agent's instructions + tools
  3. Handle function tool calls locally in a loop
  4. Cleanup agents on exit (unless keep_agents=True)
"""
import os
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

logger = logging.getLogger(__name__)

API_VERSION = "2025-05-15-preview"
RESPONSES_API_VERSION = "preview"


class _nullcontext:
    """Minimal no-op context manager for when tracer is None."""
    def __enter__(self): return None
    def __exit__(self, *args): pass


@dataclass
class RunResult:
    """Result from run_agent, including response text and tool-call stats."""
    text: str
    documents_retrieved: int = 0
    sources_called: List[str] = field(default_factory=list)


@dataclass
class AgentInfo:
    """Lightweight wrapper returned by create_agent."""
    name: str
    version: str
    id: str  # "{name}:{version}" — kept for backward compat with architectures
    instructions: str = ""
    model: str = ""


class FoundryAgentManager:
    """Manages Foundry Agent lifecycle via v2 REST API and Responses API.

    Agents are registered via the v2 REST API so they appear in the new
    Foundry portal. Invocation uses the OpenAI Responses API directly
    with the agent's instructions as a system message.
    """

    def __init__(self, govinfo_api_key: str = None, keep_agents: bool = False):
        self.endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        if not self.endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable must be set")
        self.endpoint = self.endpoint.rstrip("/")

        self.credential = DefaultAzureCredential()
        self.model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o")
        self.fast_model = os.environ.get("MODEL_DEPLOYMENT_NAME_FAST", self.model)
        self.govinfo_api_key = govinfo_api_key
        self.keep_agents = keep_agents
        self._created_agents: List[str] = []  # list of agent names
        self._agent_info: dict = {}  # agent_id -> AgentInfo
        self._tool_dispatch: dict = {}  # fn_name -> callable

        # Extract resource host for OpenAI Responses API
        # Endpoint: https://{resource}.services.ai.azure.com/api/projects/{project}
        m = re.match(r"(https://[^/]+\.services\.ai\.azure\.com)", self.endpoint)
        if not m:
            raise ValueError(f"Cannot extract resource host from endpoint: {self.endpoint}")
        resource_host = m.group(1)

        # Build OpenAI client for Responses API (needs /openai/v1/ base URL)
        from openai import AzureOpenAI
        token_provider = get_bearer_token_provider(
            self.credential, "https://cognitiveservices.azure.com/.default"
        )
        self.openai_client = AzureOpenAI(
            base_url=f"{resource_host}/openai/v1/",
            azure_ad_token_provider=token_provider,
            api_version=RESPONSES_API_VERSION,
        )

        logger.info("FoundryAgentManager (v2) initialized — endpoint=%s model=%s keep_agents=%s",
                     self.endpoint, self.model, keep_agents)

    # ── REST helpers ──────────────────────────────────────────────────

    def _get_token(self) -> str:
        return self.credential.get_token("https://ai.azure.com/.default").token

    def _rest(self, method: str, path: str, body: dict = None) -> dict:
        url = f"{self.endpoint}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }
        params = {"api-version": API_VERSION}
        resp = requests.request(method, url, headers=headers, params=params,
                                json=body, timeout=30)
        if not resp.ok:
            logger.error("REST %s %s → %d: %s", method, path, resp.status_code, resp.text)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # ── Agent creation ────────────────────────────────────────────────

    def create_agent(
        self,
        name: str,
        instructions: str,
        tools: Optional[list] = None,
        model_override: Optional[str] = None,
    ) -> AgentInfo:
        """Register a v2 Foundry Agent and track it for cleanup.

        Uses POST .../agents/{name}/versions with a prompt definition.
        The agent's instructions and model are stored locally for
        invocation via the Responses API.
        """
        model = model_override or self.model
        tool_count = len(tools) if tools else 0
        logger.info("Creating v2 agent name=%s model=%s tools=%d", name, model, tool_count)

        # Convert FunctionTool definitions to v2 format
        v2_tools = self._convert_tools(tools) if tools else []

        definition = {
            "kind": "prompt",
            "model": model,
            "instructions": instructions,
        }
        if v2_tools:
            definition["tools"] = v2_tools

        body = {"definition": definition}

        agent_name = name.lower().replace(" ", "-")
        result = self._rest("POST", f"agents/{agent_name}/versions", body)
        version = result.get("version", "1")
        agent_id = f"{agent_name}:{version}"

        info = AgentInfo(
            name=agent_name, version=version, id=agent_id,
            instructions=instructions, model=model,
        )
        self._created_agents.append(agent_name)
        self._agent_info[agent_id] = info
        logger.info("Agent created: id=%s version=%s", agent_id, version)

        return info

    def _convert_tools(self, tools: list) -> list:
        """Convert SDK FunctionTool definitions to v2 REST format.
        
        v2 expects function tools with 'name' at the top level:
        {"type": "function", "name": "...", "function": {"description": "...", "parameters": {...}}}
        """
        v2_tools = []
        for tool_def in tools:
            td = tool_def.as_dict() if hasattr(tool_def, "as_dict") else tool_def
            if isinstance(td, dict) and td.get("type") == "function":
                fn = td.get("function", {})
                v2_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "function": {
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    },
                })
            else:
                v2_tools.append(td)
        return v2_tools

    # ── Agent invocation ──────────────────────────────────────────────

    def run_agent(
        self,
        agent_id: str,
        user_message: str,
        thread_id: Optional[str] = None,
        tool_set=None,
    ) -> RunResult:
        """Run a v2 agent via the OpenAI Responses API.

        Uses the agent's stored instructions as a system message and
        passes tools for function calling. Handles the tool-call loop.
        Emits OpenTelemetry spans for agent runs and tool calls.
        """
        from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
        from utils.tracing import get_tracer

        tracer = get_tracer()

        # Look up agent info
        info = self._agent_info.get(agent_id)
        if not info:
            if ":" in agent_id:
                agent_name, _ = agent_id.rsplit(":", 1)
            else:
                agent_name = agent_id
            logger.warning("Agent info not cached for %s — running without instructions", agent_id)
            info = AgentInfo(name=agent_name, version="1", id=agent_id, model=self.model)

        logger.info("run_agent (v2): agent=%s model=%s tool_set=%s",
                     info.id, info.model,
                     type(tool_set).__name__ if tool_set else None)
        logger.debug("run_agent: message=%s", user_message[:200])

        # Register tool functions for local dispatch
        if tool_set and hasattr(tool_set, '_functions'):
            self._tool_dispatch.update(tool_set._functions)

        # Build function tool defs for Responses API
        response_tools = None
        if tool_set and hasattr(tool_set, 'definitions'):
            response_tools = []
            for td in tool_set.definitions:
                td_dict = td.as_dict() if hasattr(td, "as_dict") else td
                if isinstance(td_dict, dict) and td_dict.get("type") == "function":
                    response_tools.append({
                        "type": "function",
                        "name": td_dict["function"]["name"],
                        "description": td_dict["function"].get("description", ""),
                        "parameters": td_dict["function"].get("parameters", {}),
                    })

        # Build input with instructions as system message
        input_messages = []
        if info.instructions:
            input_messages.append({"role": "developer", "content": info.instructions})
        input_messages.append({"role": "user", "content": user_message})

        # Wrap entire agent run in a span
        span_ctx = tracer.start_as_current_span(
            f"agent.run.{info.name}",
            attributes={
                "agent.name": info.name,
                "agent.model": info.model,
                "agent.id": info.id,
                "input.length": len(user_message),
            },
        ) if tracer else _nullcontext()

        with span_ctx as agent_span:
            # Initial Responses API call
            create_kwargs = {
                "model": info.model,
                "input": input_messages,
            }
            if response_tools:
                create_kwargs["tools"] = response_tools

            response = self.openai_client.responses.create(**create_kwargs)
            logger.info("Response created: id=%s", response.id)

            # Tool-call loop
            max_rounds = 10
            total_tool_calls = 0
            for round_num in range(max_rounds):
                fn_calls = [
                    item for item in response.output
                    if isinstance(item, ResponseFunctionToolCall)
                ]
                if not fn_calls:
                    break

                logger.info("Round %d: %d function call(s)", round_num + 1, len(fn_calls))
                total_tool_calls += len(fn_calls)
                tool_results = self._execute_tool_calls(fn_calls, tool_set, tracer)

                response = self.openai_client.responses.create(
                    model=info.model,
                    input=tool_results,
                    previous_response_id=response.id,
                    tools=response_tools or [],
                )

            # Extract text from final response
            response_text = ""
            for item in response.output:
                if isinstance(item, ResponseOutputMessage):
                    for content in item.content:
                        if hasattr(content, "text"):
                            response_text += content.text

            if agent_span:
                agent_span.set_attribute("output.length", len(response_text))
                agent_span.set_attribute("tool_calls.total", total_tool_calls)
                agent_span.set_attribute("tool_call_rounds", round_num + 1 if fn_calls or round_num > 0 else 0)

            logger.info("Run complete: %d chars, %d tool calls", len(response_text), total_tool_calls)
            logger.debug("Response preview: %s", response_text[:500])

            return RunResult(text=response_text)

    def _execute_tool_calls(self, fn_calls, tool_set, tracer=None) -> list:
        """Execute function tool calls locally and return output dicts."""
        results = []
        for fc in fn_calls:
            logger.info("  Executing: %s(%s)", fc.name, fc.arguments[:200] if fc.arguments else "")

            span_ctx = tracer.start_as_current_span(
                f"tool.call.{fc.name}",
                attributes={
                    "tool.name": fc.name,
                    "tool.arguments": fc.arguments[:500] if fc.arguments else "",
                },
            ) if tracer else _nullcontext()

            with span_ctx as tool_span:
                start = time.time()
                try:
                    args = json.loads(fc.arguments) if fc.arguments else {}
                    output = self._call_tool_function(tool_set, fc.name, args)
                    elapsed = time.time() - start
                    output_str = output if isinstance(output, str) else json.dumps(output)
                    logger.info("  %s returned %d chars in %.1fs", fc.name, len(output_str), elapsed)
                    if tool_span:
                        tool_span.set_attribute("tool.status", "success")
                        tool_span.set_attribute("tool.output_length", len(output_str))
                        tool_span.set_attribute("tool.duration_ms", int(elapsed * 1000))
                    results.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": output_str,
                    })
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error("  %s FAILED: %s", fc.name, e)
                    if tool_span:
                        tool_span.set_attribute("tool.status", "error")
                        tool_span.set_attribute("tool.error", str(e))
                        tool_span.set_attribute("tool.duration_ms", int(elapsed * 1000))
                    results.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": json.dumps({"error": str(e)}),
                    })
        return results

    def _call_tool_function(self, tool_set, fn_name: str, args: dict) -> str:
        """Find and call a function by name."""
        if fn_name in self._tool_dispatch:
            return self._tool_dispatch[fn_name](**args)
        if tool_set and hasattr(tool_set, '_functions') and fn_name in tool_set._functions:
            fn = tool_set._functions[fn_name]
            self._tool_dispatch[fn_name] = fn
            return fn(**args)
        raise ValueError(f"Function {fn_name} not found in tool_set")

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self, agent_id: Optional[str] = None):
        """Delete agents via REST. Skipped when keep_agents=True."""
        if self.keep_agents:
            names = [agent_id] if agent_id else self._created_agents
            logger.info("Skipping cleanup (keep_agents=True) — agents: %s", names)
            return
        names_to_delete = [agent_id] if agent_id else list(self._created_agents)
        if names_to_delete:
            logger.info("Cleaning up %d agent(s)", len(names_to_delete))
        for name in names_to_delete:
            agent_name = name.split(":")[0] if ":" in name else name
            try:
                self._rest("DELETE", f"agents/{agent_name}")
                logger.info("Deleted agent: %s", agent_name)
            except Exception as e:
                logger.warning("Failed to delete agent %s: %s", agent_name, e)
            self._created_agents = [a for a in self._created_agents if a != agent_name]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
