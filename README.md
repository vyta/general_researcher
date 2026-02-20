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

## Behavior-driven evals

Most eval frameworks treat evaluation as verification — you build an agent, then check if it works. This project treats eval as a **design tool**. The same BDD scenarios run across all 6 architectures.

### Why BDD syntax?

```python
@scenario("AI legislation search", category="legislation")
def test_ai_legislation(s):
    s.given("a query", "What actions has Congress taken on AI policy?")
    s.when("the agent researches this query")

    # Deterministic checks — fast, no LLM needed
    s.then("completion time should be under", 20)
    s.then("the answer should mention", "artificial intelligence")
    s.then("there should be at least 3 citations", 3)

    # Process assertions via OpenTelemetry traces
    s.then("the agent should have called", "search_govinfo")
    s.then("no tool calls should have failed")

    # AI-judged quality (Azure AI Evaluation SDK + LLM judge)
    s.then("azure relevance score")
    s.then("azure coherence score")
    s.then("the answer should be", "comprehensive and well-sourced")
```

Scenarios are readable by non-engineers ("this should also check for Executive Order 14110" requires no Python knowledge), composable across assertion types, and extensible — adding a new step type is one function.

### Three layers of assertions

| Layer | What it checks | Examples | Cost |
|-------|---------------|----------|------|
| **Deterministic** | Measurable outcomes | Keyword presence, citation count, latency, document retrieval | Free, fast |
| **Process (traces)** | Intermediate agent behavior | Tool calls made, tool failures, LLM round count | Free, requires OTel |
| **AI-judged** | Subjective quality | Relevance, coherence, groundedness, fluency | LLM call per check |

The process layer is the key differentiator. An agent can produce a correct-looking answer that's entirely hallucinated because it never called the right tool. Outcome-only eval misses this entirely. By instrumenting the agent's tool-call loop with OpenTelemetry spans and asserting against them in the same scenario, you evaluate **how** the agent arrived at its answer, not just **what** it said.

### Scored metrics, not pass/fail

Every step produces a **0.0–1.0 score** mapped to one of 5 metric categories. Pass/fail is derived (steps pass at > 0.5, scenarios at ≥ 0.7), but the scores are the signal:

```
  Metric          Score        Steps
  ────────────────────────────────
  ⏱️ latency       0.85 [████████░░]  (2)
  📚 coverage      0.67 [██████░░░░]  (4)
  🎯 relevance     0.92 [█████████░]  (3)
  📎 groundedness  0.50 [█████░░░░░]  (2)
  ✨ quality        0.78 [███████░░░]  (3)
```

A binary pass/fail says "failed." Scores say "failed because groundedness is weak — the agent finds the right stuff but doesn't cite it, so add a citation-formatting step, not better retrieval."

### Dual-mode tracing

OpenTelemetry runs with two exporters simultaneously:
- **In-memory** — always active, used by the eval runner to capture and assert against spans per scenario
- **Cloud (App Insights)** — opt-in via `APPLICATIONINSIGHTS_CONNECTION_STRING`, for persisting traces from production runs

Same instrumentation serves dev-time eval and production observability.

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
