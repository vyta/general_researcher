# General Researcher - A Playground

Disclaimer: this is for funsies.

A personal playground for exploring agent architecture patterns and evaluation approaches. Uses government research as the problem domain — querying public APIs (GovInfo, Federal Register, Data.gov) through different agent compositions built on [Microsoft Foundry Agents](https://learn.microsoft.com/en-us/azure/ai-services/agents/).

## What's in here

**5 agent roles** composed by **6 architecture patterns**, plus a **BDD-style eval framework** for comparing them.

```
Agents:        Researcher · Critic · Synthesizer · Planner · Source Worker
Architectures: single_agent · researcher_critic · multi_agent
               supervisor_worker · plan_and_execute · hybrid_p2p
Eval:          BDD scenarios with scored metrics (latency/coverage/relevance/groundedness/quality)
```

## Agents

| Agent | What it does | Tools | Model |
|-------|-------------|-------|-------|
| **Researcher** | Searches sources, synthesizes findings | `search_govinfo`, `search_federal_register`, `search_datagov` | gpt-4o |
| **Critic** | Scores quality, identifies gaps | — | gpt-4o-mini |
| **Synthesizer** | Produces cited final answers | — | gpt-4o |
| **Planner** | Creates step-by-step research plans | — | gpt-4o-mini |
| **Source Worker** | Searches one source, summarizes | 1 tool | gpt-4o-mini |

## Architectures

Each is a thin orchestration layer (~50–130 LOC) over the shared agents:

| Pattern | Flow | Agents used |
|---------|------|-------------|
| **Single Agent** | Researcher does everything in one tool-calling loop | R |
| **Researcher-Critic** | Researcher → Critic feedback loop (max 3 rounds) | R, C |
| **Multi-Agent** | Researcher → Critic gate → Synthesizer | R, C, S |
| **Plan-and-Execute** | Planner → Researcher → Critic → Synthesizer | P, R, C, S |
| **Supervisor-Worker** | Planner → parallel Source Workers → Critic → Synthesizer | P, W×3, C, S |
| **Hybrid P2P** | Source Workers share discoveries across rounds → Synthesizer | W×3, S |

## Eval framework

BDD-style scenarios with 0.0–1.0 scored metrics:

```python
@scenario("AI legislation search", category="legislation")
def test_ai_legislation(s):
    s.given("a query", "What actions has Congress taken on AI policy?")
    s.when("the agent researches this query")
    s.then("completion time should be under", 60)
    s.then("the answer should mention", "artificial intelligence")
    s.then("there should be at least 3 citations", 3)
    s.then("the answer should be", "comprehensive")  # LLM-judged
```

Metrics: **latency** · **coverage** · **relevance** · **groundedness** · **quality**

Steps pass at score > 0.5, scenarios pass at overall ≥ 0.7. Output is a scorecard with per-metric breakdowns.

## Usage

```bash
cp .env.example .env   # set AZURE_AI_PROJECT_ENDPOINT + model deployments
make build
make run QUERY='AI safety federal actions'
make eval              # BDD eval (single_agent)
make eval-all          # all 6 architectures
make compare QUERY='AI legislation'  # parallel comparison
```


## Promptfoo evaluation

In addition to the BDD eval framework, the project includes a [Promptfoo](https://www.promptfoo.dev/) integration that runs the same 6 scenarios through a standardised assertion pipeline. This gives you a reproducible, CLI-driven eval with per-metric scoring and an HTML report. Uses government research as the problem domain — querying public APIs (GovInfo, Federal Register, Data.gov) through different agent compositions built on Microsoft Foundry Agents.

### How it works

```
promptfooconfig.yaml          ← scenarios, assertions, provider config
promptfoo_provider.py         ← custom Python provider wrapping SingleAgentOrchestrator
promptfoo_helpers.py          ← assertion check functions (GradingResult dicts)
.promptfoo_cache/             ← per-query metadata cache (latency, citations, sources, tool spans)
```

1. **Provider** — `promptfoo_provider.py` initialises the `SingleAgentOrchestrator`, runs the research query, and writes metadata (completion time, citation count, sources used, tool-call spans) to `.promptfoo_cache/`.
2. **Assertions** — each scenario has a list of assertions that read the cached metadata and return `{ pass, score, reason }`. Assertions are defined as YAML anchors in `promptfooconfig.yaml` and call helper functions from `promptfoo_helpers.py`.
3. **Metrics** — assertions are tagged with metric names so results can be grouped:

| Category | Assertions | Source |
|----------|-----------|--------|
| **relevance** | `icontains` keywords, `check_min_length`, `check_azure_relevance` | built-in + Azure AI |
| **latency** | `check_latency` (configurable threshold) | cached metadata |
| **groundedness** | `check_min_citations`, `check_azure_groundedness` | cached metadata + Azure AI |
| **coverage** | `check_sources_include`, `check_min_documents`, `check_tool_called`, `check_min_tool_calls` | cached metadata |
| **quality** | `check_azure_coherence`, `check_azure_fluency`, `check_llm_quality` | Azure AI + LLM judge |


### Running

```bash
# Prerequisites: Node.js >= 18, Python venv with project deps, az login
npx promptfoo@latest eval            # run all 6 scenarios
npx promptfoo@latest eval --no-cache # force re-run (skip Promptfoo cache)
npx promptfoo@latest view            # open interactive web UI on localhost
npx promptfoo@latest eval --output output.html   # export HTML report
npx promptfoo@latest eval --output output.json   # export JSON results
```

> **Note:** The GovInfo API has strict rate limits. If you see 429 errors, wait a few minutes before re-running. Promptfoo caches provider outputs by default, so subsequent runs only re-evaluate assertions unless you pass `--no-cache`.

## Project layout

```
src/
├── agents/          # Foundry Agent definitions + client lifecycle
├── architectures/   # 6 orchestration patterns
├── data_sources/    # GovInfo, Federal Register, Data.gov API clients
├── evaluation/      # BDD DSL, steps, scenarios, LLM judge, runner
├── tools/           # FunctionTool wrappers
├── main.py          # CLI
├── eval.py          # Eval CLI
└── run_architecture.py
promptfoo_provider.py    # Promptfoo custom provider (SingleAgentOrchestrator)
promptfoo_helpers.py     # Assertion helper functions for Promptfoo
promptfooconfig.yaml     # Promptfoo evaluation config (scenarios + assertions)
```

## Environment

```bash
AZURE_AI_PROJECT_ENDPOINT=https://<project>.services.ai.azure.com/api
MODEL_DEPLOYMENT_NAME=gpt-4o
MODEL_DEPLOYMENT_NAME_FAST=gpt-4o-mini
```

Requires Python 3.10+, Node.js 18+ (for Promptfoo), an Azure AI Foundry project with model deployments, and `az login`.
