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
```

## Environment

```bash
AZURE_AI_PROJECT_ENDPOINT=https://<project>.services.ai.azure.com/api
MODEL_DEPLOYMENT_NAME=gpt-4o
MODEL_DEPLOYMENT_NAME_FAST=gpt-4o-mini
```

Requires Python 3.10+, an Azure AI Foundry project with model deployments, and `az login`.
