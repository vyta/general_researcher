NERDCTL_IMAGE ?= general-researcher:latest
WORKSPACE ?= $(CURDIR)
TENANT_ID ?= mngenvmcap130646.onmicrosoft.com

# ── Build ─────────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "Building container image..."
	@nerdctl build -t $(NERDCTL_IMAGE) .

.PHONY: container
container:
	@nerdctl run --rm -it --name general-researcher-container \
		-v $(WORKSPACE):/workspace \
		-w /workspace \
		$(NERDCTL_IMAGE) bash -c "az login -t $(TENANT_ID) --use-device-code && bash"

# ── Run ───────────────────────────────────────────────────────────────

LOGGING ?= 
KEEP_AGENTS ?=

.PHONY: run
run:
	@echo "Running query: $(QUERY)"
	@nerdctl run --rm \
		-v $(WORKSPACE):/workspace \
		-w /workspace \
		--env-file .env \
		$(NERDCTL_IMAGE) bash -c "uv run src/main.py --query '$(QUERY)' $(LOGGING) $(KEEP_AGENTS)"

# ── Eval ──────────────────────────────────────────────────────────────

EVAL_ARCHITECTURES ?= single_agent

.PHONY: eval
eval:
	@echo "Running BDD evaluation..."
	@nerdctl run --rm -it \
		-v $(WORKSPACE):/workspace \
		-w /workspace \
		--env-file .env \
		$(NERDCTL_IMAGE) bash -c "uv run src/eval.py --architecture $(EVAL_ARCHITECTURES) $(LOGGING)"

.PHONY: eval-all
eval-all:
	@$(MAKE) eval EVAL_ARCHITECTURES=all

# ── Test ──────────────────────────────────────────────────────────────

.PHONY: test-sources
test-sources:
	@echo "Testing data sources (no LLM needed)..."
	@nerdctl run --rm \
		-v $(WORKSPACE):/workspace \
		-w /workspace \
		$(NERDCTL_IMAGE) bash -c "uv run tests/test_sources.py"

# ── Parallel comparison ──────────────────────────────────────────────

ARCHITECTURES ?= single_agent researcher_critic multi_agent supervisor_worker plan_execute hybrid_p2p
MAX_RESULTS ?= 5
COMPARISON_RUN_DIR ?= comparison_run

.PHONY: compare
compare:
	@echo "🚀 Starting $(words $(ARCHITECTURES)) architectures in parallel..."
	@rm -rf $(COMPARISON_RUN_DIR) && mkdir -p $(COMPARISON_RUN_DIR)
	@for arch in $(ARCHITECTURES); do \
		echo "   Starting $$arch..."; \
		nerdctl run -d \
			--name general-researcher-$$arch \
			-v $(WORKSPACE):/workspace \
			-w /workspace \
			--env-file .env \
			$(NERDCTL_IMAGE) bash -c "uv run -- src/run_architecture.py -a $$arch -q '$(QUERY)' -o $(COMPARISON_RUN_DIR) --max-results $(MAX_RESULTS) $(LOGGING) $(KEEP_AGENTS)" & \
	done; \
	wait
	@echo "✅ All containers started. Check: make compare-status"

.PHONY: compare-eval
compare-eval:
	@echo "🚀 Starting $(words $(ARCHITECTURES)) architectures in parallel (eval mode)..."
	@rm -rf $(COMPARISON_RUN_DIR) && mkdir -p $(COMPARISON_RUN_DIR)
	@for arch in $(ARCHITECTURES); do \
		echo "   Starting $$arch..."; \
		nerdctl run -d \
			--name general-researcher-$$arch \
			-v $(WORKSPACE):/workspace \
			-w /workspace \
			--env-file .env \
			$(NERDCTL_IMAGE) bash -c "uv run -- src/run_architecture.py -a $$arch --eval -o $(COMPARISON_RUN_DIR) --max-results $(MAX_RESULTS) $(LOGGING) $(KEEP_AGENTS)" & \
	done; \
	wait
	@echo "✅ All containers started. Check: make compare-status"

.PHONY: compare-status
compare-status:
	@echo "📊 Comparison status:"
	@echo "  In progress:"; ls $(COMPARISON_RUN_DIR)/*_in_progress 2>/dev/null || echo "    (none)"
	@echo "  Complete:";    ls $(COMPARISON_RUN_DIR)/*_complete.json 2>/dev/null || echo "    (none)"
	@echo "  Failed:";      ls $(COMPARISON_RUN_DIR)/*_failed.json 2>/dev/null || echo "    (none)"

.PHONY: compare-stop
compare-stop:
	@for arch in $(ARCHITECTURES); do \
		nerdctl stop general-researcher-$$arch 2>/dev/null || true; \
	done
	@echo "✅ All containers stopped"

# ── Help ──────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "General Researcher — Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make build                Build container image"
	@echo "  make container            Interactive container with Azure creds"
	@echo ""
	@echo "Run:"
	@echo "  make run QUERY='...'      Run a single query"
	@echo "  make test-sources         Test data source APIs (no LLM)"
	@echo ""
	@echo "Eval:"
	@echo "  make eval                 Run BDD eval (single_agent)"
	@echo "  make eval-all             Run BDD eval (all architectures)"
	@echo ""
	@echo "Compare (parallel containers):"
	@echo "  make compare QUERY='...'  Compare architectures on a query"
	@echo "  make compare-eval         Compare architectures on eval scenarios"
	@echo "  make compare-status       Check running comparison"
	@echo "  make compare-stop         Stop all containers"
	@echo ""
	@echo "Options:"
	@echo "  ARCHITECTURES='a b'       Subset of architectures"
	@echo "  MAX_RESULTS=10            Results per source (default: 5)"
	@echo "  LOGGING=--verbose         Logging level"
	@echo "  KEEP_AGENTS=--keep-agents Keep agents in Foundry after run"
