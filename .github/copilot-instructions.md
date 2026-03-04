# Copilot Instructions — General Researcher

## Architecture overview

BDD eval framework comparing agent architectures for research tasks. Two runtime modes: **Foundry agents** (Azure AI) and **ACP agents** (Copilot CLI via Agent Client Protocol).

```
src/
├── agents/          # Agent configs + ACP SDK wrapper (sync bridge over async SDK)
├── architectures/   # Orchestration patterns, each with research() → ResearchResult
├── data_sources/    # API clients (GovInfo, Federal Register, Data.gov)
├── evaluation/      # BDD DSL, action/step registries, runner, LLM judge
├── tools/           # FunctionTool wrappers for Foundry agents
├── utils/           # OTel dual-mode tracing
├── main.py          # CLI entry — sys.path.insert(0, src/)
└── eval.py          # Eval CLI entry — same sys.path trick
```

## Import conventions

Entry points (`main.py`, `eval.py`) add `src/` to `sys.path`. All imports are **absolute from `src/`** without a `src.` prefix. Within packages, use relative imports.

```python
# From entry points or top-level modules:
from architectures import ARCHITECTURES
from evaluation.dsl import get_all_scenarios

# Within a package (e.g., inside evaluation/):
from .dsl import StepResult
```

## Extension points

| To add…              | Where                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Architecture         | Class in `src/architectures/`, entry in `ARCHITECTURES` dict in `__init__.py` |
| Eval template        | `@template("name", category="arch_key")` in `scenarios.py`, call `.cases(dataset)` |
| One-off scenario     | `@scenario("name", category="arch_key")` in `scenarios.py`           |
| Step grader          | `@step("pattern")` in `steps.py` — matched by longest prefix         |
| Action handler       | `@action("when text")` in `actions.py`                                |
| Data source          | Subclass `DataSource` in `src/data_sources/`, add to `get_all_sources()` |

## Eval DSL patterns

Scenarios use Given/When/Then with multi-stage evaluation. Each `when()` starts a new stage; `then()` calls bind to the current stage.

```python
@template("my research", category="single_agent")
def my_template(s, data):
    s.given("a query", data["query"])
    s.when("the agent researches this query")          # Stage 1
    s.then("the agent should have called", "search_govinfo")
    s.when("the agent synthesizes the results")        # Stage 2
    s.then("the answer should mention", "keyword")
    s.then("the answer should be", data["quality_criteria"])  # LLM-judged

my_template.cases([{"query": "...", "quality_criteria": "..."}])
```

`scenarios.py` **must** be imported for registration — `eval.py` does `import evaluation.scenarios  # noqa: F401`.

Step graders return `StepResult(score=0.0–1.0, ...)`. Mark with `is_llm=True` or `is_azure_eval=True` if they need external judges; missing dependencies auto-skip with score 1.0.

## Data containers

Use `@dataclass` (not Pydantic). Key types: `ResearchResult` (architecture output), `StepResult` / `ScenarioResult` (eval results), `ACPAgentConfig` (connection config), `ResearchOutput` (normalized SUT output).

## ACP client

`ACPClient` in `src/agents/acp_client.py` wraps the async `agent-client-protocol` SDK with a sync bridge (`asyncio.new_event_loop()`). Two transports: `stdio` (spawns subprocess) and `tcp` (connects to running server). The `_EvalClient` callback auto-approves permission requests and accumulates response text.

## Commands

```bash
uv sync                                          # install deps
uv run src/main.py --query "..."                  # single query
uv run src/eval.py -a acp_agent --no-azure-eval -v  # eval with answer preview
uv run src/eval.py -a single_agent --no-llm-judge --no-azure-eval  # fast dev
make build && make container                      # containerized run
```

## Conventions

- Architecture classes expose `research(query, max_results_per_source=5) → ResearchResult`
- Registry keys and template categories use `snake_case` matching the architecture key
- Private registry functions prefixed with `_` (e.g., `_research`, `_answer_should_mention`)
- Module-level docstrings on every file describing purpose
- `field(default_factory=list)` for mutable dataclass defaults
- `verbose` flag gates answer output in both terminal and JSON results
