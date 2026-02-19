"""
Eval scenarios for the government research agent.

Each scenario is a BDD-style test that defines:
  - Given: the research query
  - When: the agent processes it
  - Then: assertions about the output (deterministic + LLM-judged)

Import this module to register all scenarios before running evals.
"""
from .dsl import scenario

# latency constant
MAX_COMPLETION_TIME = 20  # seconds

# ── Legislation ───────────────────────────────────────────────────────

@scenario("AI legislation search", category="legislation")
def test_ai_legislation(s):
    s.given("a query", "What actions has Congress taken on artificial intelligence policy?")
    s.when("the agent researches this query")
    s.then("completion time should be under", MAX_COMPLETION_TIME)
    s.then("the answer should mention", "artificial intelligence")
    s.then("the answer should mention", "Congress")
    s.then("there should be at least 3 citations", 3)
    s.then("the answer should be at least 200 characters", 200)
    s.then("sources should include", "GovInfo")
    s.then("documents retrieved should be at least", 3)
    s.then("azure relevance score")
    s.then("azure coherence score")
    s.then("azure groundedness score")
    s.then("azure fluency score")
    s.then("the answer should be", "comprehensive and well-sourced")


@scenario("Climate and energy legislation", category="legislation")
def test_climate_energy(s):
    s.given("a query", "Legislation related to climate change and renewable energy")
    s.when("the agent researches this query")
    s.then("completion time should be under", MAX_COMPLETION_TIME)
    s.then("the answer should mention", "climate")
    s.then("the answer should mention", "energy")
    s.then("there should be at least 2 citations", 2)
    s.then("the answer should be at least 150 characters", 150)
    s.then("documents retrieved should be at least", 3)
    s.then("the answer should be", "factual and grounded in cited sources")


# ── Regulations ───────────────────────────────────────────────────────

@scenario("EPA water regulations", category="regulations")
def test_epa_water(s):
    s.given("a query", "EPA regulations on clean water standards")
    s.when("the agent researches this query")
    s.then("completion time should be under", MAX_COMPLETION_TIME)
    s.then("the answer should mention", "water")
    s.then("the answer should mention", "EPA")
    s.then("there should be at least 2 citations", 2)
    s.then("the answer should be at least 150 characters", 150)
    s.then("sources should include", "Federal Register")
    s.then("documents retrieved should be at least", 3)
    s.then("azure relevance score")
    s.then("azure coherence score")
    s.then("azure groundedness score")
    s.then("the answer should be", "specific about regulatory details")


@scenario("Cybersecurity regulations", category="regulations")
def test_cybersecurity_regs(s):
    s.given("a query", "What federal regulations were issued regarding cybersecurity?")
    s.when("the agent researches this query")
    s.then("completion time should be under", MAX_COMPLETION_TIME)
    s.then("the answer should mention", "cybersecurity")
    s.then("there should be at least 2 citations", 2)
    s.then("the answer should be at least 150 characters", 150)
    s.then("documents retrieved should be at least", 2)
    s.then("the answer should be", "comprehensive about federal cybersecurity requirements")


# ── Datasets ──────────────────────────────────────────────────────────

@scenario("Public health datasets", category="datasets")
def test_public_health_data(s):
    s.given("a query", "What government datasets are available about public health?")
    s.when("the agent researches this query")
    s.then("completion time should be under", MAX_COMPLETION_TIME)
    s.then("the answer should mention", "health")
    s.then("the answer should mention", "data")
    s.then("there should be at least 2 citations", 2)
    s.then("the answer should be at least 150 characters", 150)
    s.then("sources should include", "Data.gov")
    s.then("documents retrieved should be at least", 3)
    s.then("the answer should be", "informative about available data resources")


# ── Policy ────────────────────────────────────────────────────────────

@scenario("Immigration policy changes", category="policy")
def test_immigration_policy(s):
    s.given("a query", "Recent immigration policy changes and border security")
    s.when("the agent researches this query")
    s.then("completion time should be under", MAX_COMPLETION_TIME)
    s.then("the answer should mention", "immigration")
    s.then("the answer should mention", "border")
    s.then("there should be at least 2 citations", 2)
    s.then("the answer should be at least 150 characters", 150)
    s.then("documents retrieved should be at least", 2)
    s.then("the answer should be", "balanced and evidence-based")
