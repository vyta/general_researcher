"""
Eval scenarios for the government research agent.

Architecture-specific templates define grader behavior for each pipeline pattern.
Datasets provide the queries. Each dataset entry generates a case per template.

Templates model the actual pipeline stages of each architecture:
  - single_agent: research → synthesize
  - single_agent_code: research → execute code → synthesize
  - researcher_critic: research → critic feedback → synthesize
  - multi_agent: research → critic gate → synthesize
  - plan_and_execute: plan → execute → critic → synthesize
  - supervisor_worker: plan → parallel workers → critic → synthesize
  - hybrid_p2p: parallel workers with peer sharing → synthesize

Import this module to register all scenarios before running evals.
"""
from .dsl import scenario, template

MAX_COMPLETION_TIME = 20  # seconds
ACP_TIMEOUT = 120  # ACP agents may be slower (subprocess + MCP tools)


# ═══════════════════════════════════════════════════════════════════════
# Assertion composition helper
# ═══════════════════════════════════════════════════════════════════════

class AssertionGroup:
    """
    Helper for composing common assertion combinations in scenario templates.
    
    Consolidates frequently repeated assertion patterns into reusable methods,
    reducing code duplication across architecture templates while maintaining
    identical assertion behavior.
    
    Usage:
        @template("my template", category="my_cat")
        def my_template(s, data):
            s.given("a query", data["query"])
            s.when("the agent researches")
            AssertionGroup(s).research_expectations(data)
            s.when("the agent synthesizes")
            AssertionGroup(s).synthesis_expectations(data)
    """
    
    def __init__(self, scenario_builder):
        """
        Args:
            scenario_builder: ScenarioBuilder instance (the 's' parameter in templates).
        """
        self.s = scenario_builder
    
    def research_expectations(self, data, include_search_queries=True):
        """
        Common research phase assertions for document retrieval and tool usage.
        
        Includes: expected source calls, tool call validation, document counts,
        source diversity, and search query metrics.
        
        Args:
            data: Scenario data dict with keys like expected_source, min_docs, etc.
            include_search_queries: Whether to assert minimum search queries (default True)
        """
        if data.get("expected_source"):
            self.s.then("the agent should have called", data["expected_source"])
            self.s.then("sources should include", data["source_label"])
        self.s.then("no tool calls should have failed")
        self.s.then("total tool calls should be at least", 1)
        self.s.then("no redundant tool calls")
        self.s.then("documents retrieved should be at least", data.get("min_docs", 3))
        self.s.then("unique sources used should be at least", data.get("min_sources", 1))
        if include_search_queries and data.get("min_search_queries"):
            self.s.then("search queries should be at least", data["min_search_queries"])
    
    def synthesis_expectations(self, data, include_numbers=False):
        """
        Common synthesis/answer phase assertions for answer quality.
        
        Includes: keyword mentions, temporal term matching, citation counts,
        minimum length, and LLM-judged quality criteria.
        
        Args:
            data: Scenario data dict with expected_terms, quality_criteria, etc.
            include_numbers: Whether to assert a number appears in answer (default False)
        """
        for term in data["expected_terms"]:
            self.s.then("the answer should mention", term)
        if data.get("temporal_terms"):
            self.s.then("the answer should mention one of", data["temporal_terms"])
        if include_numbers:
            self.s.then("the answer should contain a number")
        self.s.then("there should be at least 2 citations", data.get("min_citations", 2))
        self.s.then("the answer should be at least 150 characters", data.get("min_length", 150))
        self.s.then("the answer should be", data["quality_criteria"])
    
    def critic_expectations(self):
        """
        Common critic evaluation assertions.
        
        Validates that critic agent ran and respects iteration limits.
        """
        self.s.then("the critic should have run")
        self.s.then("critic iterations should be at most", 3)
    
    def code_execution_expectations(self):
        """
        Code execution phase assertions.
        
        Validates that code was executed without errors.
        """
        self.s.then("code should have been executed")
        self.s.then("no code execution errors")


# ═══════════════════════════════════════════════════════════════════════
# Architecture-specific templates
# ═══════════════════════════════════════════════════════════════════════

# ── Single Agent ──────────────────────────────────────────────────────

@template("single agent", category="single_agent")
def single_agent(s, data):
    """Baseline: one researcher agent does everything."""
    s.given("a query", data["query"])

    s.when("the agent researches this query")
    AssertionGroup(s).research_expectations(data)

    s.when("the agent synthesizes the results")
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME))
    AssertionGroup(s).synthesis_expectations(data)


# ── Single Agent + Code Execution ────────────────────────────────────

@template("single agent code", category="single_agent_code")
def single_agent_code(s, data):
    """Researcher with code execution for quantitative analysis."""
    s.given("a query", data["query"])

    s.when("the agent researches this query")
    AssertionGroup(s).research_expectations(data)

    s.when("the agent executes code")
    AssertionGroup(s).code_execution_expectations()

    s.when("the agent synthesizes the results")
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME * 2))
    AssertionGroup(s).synthesis_expectations(data, include_numbers=True)


# ── Researcher-Critic ────────────────────────────────────────────────

@template("researcher critic", category="researcher_critic")
def researcher_critic(s, data):
    """Researcher → Critic feedback loop (max 3 rounds)."""
    s.given("a query", data["query"])

    s.when("the agent researches this query")
    AssertionGroup(s).research_expectations(data)

    s.when("the critic evaluates the research")
    AssertionGroup(s).critic_expectations()

    s.when("the agent synthesizes the results")
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME * 2))
    AssertionGroup(s).synthesis_expectations(data)


# ── Multi-Agent (R → C → S) ─────────────────────────────────────────

@template("multi agent", category="multi_agent")
def multi_agent(s, data):
    """Researcher → Critic gate → Synthesizer."""
    s.given("a query", data["query"])

    s.when("the agent researches this query")
    AssertionGroup(s).research_expectations(data)

    s.when("the critic evaluates the research")
    AssertionGroup(s).critic_expectations()
    s.then("distinct agents should have run at least", 3)

    s.when("the synthesizer produces the answer")
    s.then("the synthesizer should have run")
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME * 2))
    AssertionGroup(s).synthesis_expectations(data)


# ── Plan-and-Execute ─────────────────────────────────────────────────

@template("plan and execute", category="plan_execute")
def plan_execute(s, data):
    """Planner → Researcher → Critic → Synthesizer with replanning."""
    s.given("a query", data["query"])

    s.when("the agent plans the research")
    s.then("the planner should have run")

    s.when("the agent executes the plan")
    AssertionGroup(s).research_expectations(data)

    s.when("the critic evaluates the research")
    AssertionGroup(s).critic_expectations()

    s.when("the synthesizer produces the answer")
    s.then("the synthesizer should have run")
    s.then("distinct agents should have run at least", 4)
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME * 3))
    AssertionGroup(s).synthesis_expectations(data)


# ── Supervisor-Worker ────────────────────────────────────────────────

@template("supervisor worker", category="supervisor_worker")
def supervisor_worker(s, data):
    """Planner → parallel Source Workers → Critic → Synthesizer."""
    s.given("a query", data["query"])

    s.when("the supervisor plans the research")
    s.then("the planner should have run")

    s.when("the source workers execute")
    s.then("source workers should have run at least", data.get("min_workers", 2))
    # Use research_expectations without expected_source (handled by workers)
    s.then("no tool calls should have failed")
    s.then("no redundant tool calls")
    s.then("documents retrieved should be at least", data.get("min_docs", 3))
    s.then("unique sources used should be at least", data.get("min_sources", 2))
    if data.get("min_search_queries"):
        s.then("search queries should be at least", data["min_search_queries"])

    s.when("the critic evaluates the research")
    s.then("the critic should have run")

    s.when("the synthesizer produces the answer")
    s.then("the synthesizer should have run")
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME * 3))
    AssertionGroup(s).synthesis_expectations(data)


# ── Hybrid P2P ───────────────────────────────────────────────────────

@template("hybrid p2p", category="hybrid_p2p")
def hybrid_p2p(s, data):
    """Source Workers share discoveries across rounds → Synthesizer."""
    s.given("a query", data["query"])

    s.when("the workers research with peer sharing")
    s.then("source workers should have run at least", data.get("min_workers", 2))
    s.then("no tool calls should have failed")
    s.then("no redundant tool calls")
    s.then("documents retrieved should be at least", data.get("min_docs", 3))
    s.then("unique sources used should be at least", data.get("min_sources", 2))
    if data.get("min_search_queries"):
        s.then("search queries should be at least", data["min_search_queries"])

    s.when("the synthesizer produces the answer")
    s.then("the synthesizer should have run")
    s.then("completion time should be under", data.get("max_time", MAX_COMPLETION_TIME * 3))
    AssertionGroup(s).synthesis_expectations(data)


# ── ACP Agent ─────────────────────────────────────────────────────────

@template("acp agent", category="acp_agent")
def acp_agent(s, data):
    """ACP agent: answer quality only (internals are opaque)."""
    s.given("a query", data["query"])

    s.when("the agent researches this query")
    # No tool-introspection assertions — ACP agent internals are opaque

    s.when("the agent synthesizes the results")
    s.then("completion time should be under", data.get("max_time", ACP_TIMEOUT))
    for term in data["expected_terms"]:
        s.then("the answer should mention", term)
    if data.get("min_citations", 0) > 0:
        s.then("there should be at least 2 citations", data.get("min_citations", 1))
    s.then("the answer should be at least 150 characters", data.get("min_length", 100))
    s.then("the answer should be", data["quality_criteria"])


LEGISLATION_CASES = [
    {
        "query": "What actions has Congress taken on artificial intelligence policy?",
        "expected_terms": ["artificial intelligence", "Congress"],
        "quality_criteria": "comprehensive and well-sourced",
        "expected_source": "search_govinfo",
        "source_label": "GovInfo",
        "min_citations": 3,
        "min_length": 200,
    },
    {
        "query": "Legislation related to climate change and renewable energy",
        "expected_terms": ["climate", "energy"],
        "quality_criteria": "factual and grounded in cited sources",
    },
]

REGULATION_CASES = [
    {
        "query": "EPA regulations on clean water standards",
        "expected_terms": ["water", "EPA"],
        "quality_criteria": "specific about regulatory details",
        "expected_source": "search_federal_register",
        "source_label": "Federal Register",
        "min_docs": 3,
    },
    {
        "query": "What federal regulations were issued regarding cybersecurity?",
        "expected_terms": ["cybersecurity"],
        "quality_criteria": "comprehensive about federal cybersecurity requirements",
    },
]

DATASET_CASES = [
    {
        "query": "What government datasets are available about public health?",
        "expected_terms": ["health", "data"],
        "quality_criteria": "informative about available data resources",
        "expected_source_label": "Data.gov",
    },
]

POLICY_CASES = [
    {
        "query": "Recent immigration policy changes and border security",
        "expected_terms": ["immigration", "border"],
        "quality_criteria": "balanced and evidence-based",
    },
]

ANALYTICAL_CASES = [
    {
        "query": "Compare the number of bills introduced related to AI versus cybersecurity in the most recent Congress. Which topic had more legislative activity?",
        "expected_terms": ["bill"],
        "quality_criteria": "a quantitative comparison with specific counts, not just qualitative statements",
        "expected_source": "search_govinfo",
        "source_label": "GovInfo",
        "min_docs": 4,
    },
    {
        "query": "How many final rules has the EPA published related to air quality in the past year? Summarize the trend.",
        "expected_terms": ["EPA"],
        "quality_criteria": "data-driven with specific counts or percentages, not vague generalizations",
        "expected_source": "search_federal_register",
        "source_label": "Federal Register",
        "min_docs": 2,
    },
    {
        "query": "What government datasets exist for tracking federal spending, and how many legislative actions reference budget transparency? Provide a breakdown.",
        "expected_terms": ["budget"],
        "quality_criteria": "a structured breakdown combining dataset availability with legislative context",
        "min_citations": 3,
        "min_sources": 2,
    },
]

AMBIGUOUS_CASES = [
    {
        "query": "AI policy",
        "expected_terms": ["artificial intelligence"],
        "quality_criteria": "specific about which jurisdiction or policy area, not a vague overview",
        "min_citations": 2,
    },
    {
        "query": "water rules",
        "expected_terms": ["water"],
        "quality_criteria": "specific about which regulations or standards, not generic",
        "min_citations": 2,
    },
]

MULTI_PART_CASES = [
    {
        "query": "What has Congress done on AI policy and how does it compare to cybersecurity legislation?",
        "expected_terms": ["artificial intelligence", "cybersecurity"],
        "quality_criteria": "addresses both AI and cybersecurity with explicit comparison between them",
        "min_citations": 3,
        "min_sources": 2,
    },
    {
        "query": "What EPA regulations exist for water quality and what datasets track compliance?",
        "expected_terms": ["EPA", "water"],
        "quality_criteria": "addresses both the regulatory framework and available data sources separately",
        "min_citations": 2,
        "min_sources": 2,
    },
]

TEMPORAL_CASES = [
    {
        "query": "What are the most recent changes to clean water regulations?",
        "expected_terms": ["water", "regulation"],
        "temporal_terms": ["2024", "2025", "recent"],
        "quality_criteria": "focuses on recent regulatory changes, not a historical overview",
    },
    {
        "query": "Latest congressional action on AI in the current session",
        "expected_terms": ["AI", "Congress"],
        "temporal_terms": ["2025", "118th", "119th", "current"],
        "quality_criteria": "specific to the current or most recent congressional session, not a history lesson",
    },
]

SOURCE_SELECTION_CASES = [
    {
        "query": "What final rules has the EPA published on emissions standards?",
        "expected_terms": ["EPA", "emissions"],
        "quality_criteria": "specific about regulatory details from the Federal Register",
        "expected_source": "search_federal_register",
        "source_label": "Federal Register",
        "min_docs": 3,
    },
    {
        "query": "Congressional hearings on technology antitrust",
        "expected_terms": ["antitrust"],
        "quality_criteria": "specific about congressional activity, not general news",
        "expected_source": "search_govinfo",
        "source_label": "GovInfo",
    },
]

MULTI_SOURCE_CASES = [
    {
        "query": "How do federal regulations on drug pricing compare to legislative proposals in Congress?",
        "expected_terms": ["drug", "pricing"],
        "quality_criteria": "synthesizes both regulatory and legislative perspectives with explicit comparison",
        "min_sources": 2,
        "min_search_queries": 2,
        "min_citations": 3,
    },
    {
        "query": "What government datasets track air quality, and what EPA rules govern air pollution?",
        "expected_terms": ["air quality"],
        "quality_criteria": "covers both available datasets and regulatory requirements separately",
        "min_sources": 2,
        "min_search_queries": 2,
        "min_citations": 2,
    },
]

MATERIAL_SUBSTITUTION_CASES = [
    {
        "query": "What materials can substitute for O-ring 10001?",
        "expected_terms": ["substitute", "O-ring"],
        "quality_criteria": (
            "lists valid substitute materials with confidence scores "
            "and explains context-dependent constraints"
        ),
        "min_length": 100,
    },
    {
        "query": "List all slow-moving inventory items.",
        "expected_terms": ["slow-moving"],
        "quality_criteria": (
            "returns a structured list of materials flagged as "
            "slow-moving with on-hand quantities"
        ),
        "min_length": 80,
    },
    {
        "query": (
            "Can I use material 10003 (NBR) instead of 10001 (Viton) "
            "in the WIX 51515 filter?"
        ),
        "expected_terms": ["substitut"],  # matches substitute/substitution
        "quality_criteria": (
            "addresses material compatibility with specific reasoning, "
            "mentions temperature or operating constraints if applicable"
        ),
        "min_length": 150,
    },
]

# Merge all cases into one dataset
ALL_CASES = (LEGISLATION_CASES + REGULATION_CASES + DATASET_CASES + POLICY_CASES
             + AMBIGUOUS_CASES + MULTI_PART_CASES + TEMPORAL_CASES
             + SOURCE_SELECTION_CASES + MULTI_SOURCE_CASES)

# ═══════════════════════════════════════════════════════════════════════
# Generate cases: each architecture × each query
# ═══════════════════════════════════════════════════════════════════════

# Every architecture gets all cases (including query interpretation)
single_agent.cases(ALL_CASES)
researcher_critic.cases(ALL_CASES)
multi_agent.cases(ALL_CASES)
plan_execute.cases(ALL_CASES)
supervisor_worker.cases(ALL_CASES)
hybrid_p2p.cases(ALL_CASES)

# Code execution architecture gets analytical cases (quantitative queries)
single_agent_code.cases(ANALYTICAL_CASES)

# ACP agent gets domain-specific material-substitution cases
acp_agent.cases(MATERIAL_SUBSTITUTION_CASES)
