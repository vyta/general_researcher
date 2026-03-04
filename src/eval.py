"""
Run BDD-style evaluation suite on architectures.
"""
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow absolute imports from src
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import nullcontext

from data_sources import get_all_sources
from agents.client import FoundryAgentManager
from architectures import ARCHITECTURES

# Import scenarios to register them
import evaluation.scenarios  # noqa: F401
from evaluation.dsl import get_all_scenarios
from evaluation.runner import EvalRunner
from evaluation.llm_judge import LLMJudge
from utils.tracing import setup_tracing


def main():
    load_dotenv()
    setup_tracing()
    
    parser = argparse.ArgumentParser(description="Evaluate research architectures (BDD-style)")
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        nargs="+",
        choices=list(ARCHITECTURES.keys()) + ["all"],
        default=["single_agent"],
        help="Architecture(s) to evaluate (default: single_agent)"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Only run scenarios in this category"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=None,
        help="Only run a specific scenario by id"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-judged assertions (faster, deterministic only)"
    )
    parser.add_argument(
        "--no-azure-eval",
        action="store_true",
        help="Disable Azure AI Evaluation SDK evaluators"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--keep-agents",
        action="store_true",
        help="Keep agents in Foundry after evaluation"
    )
    parser.add_argument(
        "--acp-cwd",
        type=str,
        default=None,
        help="Working directory for the ACP agent subprocess"
    )
    parser.add_argument(
        "--acp-transport",
        type=str,
        choices=["stdio", "tcp"],
        default="stdio",
        help="ACP transport mode (default: stdio)"
    )
    parser.add_argument(
        "--acp-executable",
        type=str,
        default="copilot",
        help="ACP agent executable for stdio transport (default: copilot)"
    )
    parser.add_argument(
        "--acp-host",
        type=str,
        default="localhost",
        help="ACP server hostname for TCP transport (default: localhost)"
    )
    parser.add_argument(
        "--acp-port",
        type=int,
        default=3000,
        help="ACP server port for TCP transport (default: 3000)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    
    # Determine which architectures to evaluate
    if "all" in args.architecture:
        # Exclude external (ACP) architectures from "all" — they need a running server
        arch_keys = [k for k, v in ARCHITECTURES.items() if not v.get("external")]
    else:
        arch_keys = args.architecture
    
    # Filter scenarios
    scenarios = get_all_scenarios()
    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
    if args.scenario:
        scenarios = [s for s in scenarios if s.scenario_id == args.scenario]
    
    if not scenarios:
        print("No scenarios found matching filters.")
        return
    
    print("=" * 70)
    print("  GENERAL RESEARCHER — BEHAVIORAL EVALUATION")
    print("=" * 70)
    print(f"  Architectures: {', '.join(arch_keys)}")
    print(f"  Total scenarios: {len(scenarios)}")
    if args.no_llm_judge:
        print("  LLM judge: disabled")
    if args.no_azure_eval:
        print("  Azure evaluators: disabled")
    
    # Initialize shared components — only create Foundry manager if needed
    needs_foundry = any(
        not ARCHITECTURES[k].get("external") for k in arch_keys
    )

    manager = None
    data_sources = None
    if needs_foundry:
        data_sources = get_all_sources()
        manager = FoundryAgentManager(keep_agents=args.keep_agents)
    
    # Build LLM judge if enabled
    llm_judge = None
    if not args.no_llm_judge and manager is not None:
        llm_judge = LLMJudge(manager.openai_client, model=manager.fast_model)
    elif not args.no_llm_judge:
        # Fallback: create AzureOpenAI client from env without full Foundry manager
        try:
            import re
            from openai import AzureOpenAI
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").rstrip("/")
            if not endpoint:
                raise ValueError("AZURE_AI_PROJECT_ENDPOINT not set")
            # AI Foundry: https://{resource}.services.ai.azure.com/...
            # Classic Azure OpenAI: https://{resource}.openai.azure.com/...
            m = re.match(r"(https://[^/]+\.(?:services\.ai|openai)\.azure\.com)", endpoint)
            if not m:
                raise ValueError(f"Unrecognized endpoint format: {endpoint}")
            resource_host = m.group(1)
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client = AzureOpenAI(
                azure_endpoint=resource_host,
                azure_ad_token_provider=token_provider,
                api_version="2025-03-01-preview",
            )
            model = os.environ.get("MODEL_DEPLOYMENT_NAME_FAST",
                                   os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o"))
            llm_judge = LLMJudge(client, model=model)
            print(f"  LLM judge: enabled (Azure OpenAI, {model})")
        except Exception as e:
            print(f"  LLM judge: skipped ({e})")
    
    # Build Azure evaluators if enabled
    azure_evaluators = None
    if not args.no_azure_eval and manager is not None:
        try:
            from evaluation.azure_evaluators import AzureEvaluators
            azure_evaluators = AzureEvaluators.from_env(credential=manager.credential)
            print("  Azure evaluators: enabled")
        except Exception as e:
            print(f"  Azure evaluators: unavailable ({e})")

    runner = EvalRunner(output_dir=args.output_dir, llm_judge=llm_judge,
                        azure_evaluators=azure_evaluators, verbose=args.verbose)
    
    all_results = {}
    acp_orchestrators = []  # track for cleanup
    
    with manager if manager else nullcontext():
        for arch_key in arch_keys:
            arch_info = ARCHITECTURES[arch_key]
            print(f"\n{'=' * 70}")
            print(f"  {arch_info.get('emoji', '🔬')} {arch_info['name']}")
            print(f"{'=' * 70}")
            
            # Filter scenarios to match this architecture's category
            arch_scenarios = [s for s in scenarios if s.category == arch_key]
            if not arch_scenarios:
                print(f"  No scenarios for architecture '{arch_key}', skipping.")
                continue
            
            print(f"  Scenarios: {len(arch_scenarios)}")
            
            if arch_info.get("external"):
                from agents.smart_inventory_advisor import ACPAgentConfig
                acp_config = ACPAgentConfig(
                    name=arch_info["name"],
                    transport=args.acp_transport,
                    executable=args.acp_executable,
                    host=args.acp_host,
                    port=args.acp_port,
                    cwd=args.acp_cwd,
                )
                architecture = arch_info["class"](acp_config=acp_config)
                acp_orchestrators.append(architecture)
            else:
                architecture = arch_info["class"](manager, data_sources)
            results = runner.run_all(architecture, arch_key, scenarios=arch_scenarios)
            
            runner.print_summary(results, arch_key)
            runner.save_results(results, arch_key)
            all_results[arch_key] = results
    
    # Cleanup ACP connections
    for orch in acp_orchestrators:
        try:
            orch.close()
        except Exception:
            pass
    
    # Print comparison if multiple architectures
    if len(arch_keys) > 1:
        _print_comparison(all_results)
    
    print("\n✓ Evaluation complete!")


def _print_comparison(all_results: dict):
    """Print comparison table across architectures with metric scores."""
    from evaluation.dsl import METRIC_CATEGORIES

    print(f"\n{'=' * 70}")
    print("  📊 ARCHITECTURE COMPARISON")
    print(f"{'=' * 70}")
    
    # Header
    metrics = ["latency", "coverage", "relevance", "groundedness", "quality"]
    metric_hdrs = "".join(f"{m[:5]:>7}" for m in metrics)
    print(f"\n  {'Architecture':<22} {'Score':>6} {'P/F':>5}{metric_hdrs}")
    print(f"  {'-' * (36 + 7 * len(metrics))}")
    
    for arch_key, results in all_results.items():
        successful = [r for r in results if not r.error]
        overall = sum(r.overall_score for r in successful) / len(successful) if successful else 0.0
        passed = sum(1 for r in results if r.passed)
        
        # Per-metric averages
        metric_avgs = {}
        for r in successful:
            for s in r.steps:
                cat = s.metric or "other"
                metric_avgs.setdefault(cat, []).append(s.score)
        
        name = ARCHITECTURES[arch_key]["name"][:21]
        metric_vals = "".join(
            f"{sum(metric_avgs.get(m, [0])) / len(metric_avgs.get(m, [1])):>7.2f}" if m in metric_avgs else f"{'—':>7}"
            for m in metrics
        )
        print(f"  {name:<22} {overall:>5.2f} {passed:>2}/{len(results):<2}{metric_vals}")


if __name__ == "__main__":
    main()
