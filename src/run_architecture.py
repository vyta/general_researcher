#!/usr/bin/env python3
"""
Run a single architecture and write results to a status file.
Used for parallel container execution.

Supports two modes:
  - Query mode: Run a single query (--query "...")
  - Eval mode: Run all eval fixtures (--eval)

Creates:
  - {output_dir}/{arch}_in_progress  (while running)
  - {output_dir}/{arch}_complete.json (on success)
  - {output_dir}/{arch}_failed.json (on error)
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from utils.tracing import setup_tracing
setup_tracing()

from data_sources import get_all_sources
from agents.client import FoundryAgentManager
from architectures import ARCHITECTURES
from evaluation.dsl import get_all_scenarios
from evaluation.runner import EvalRunner, ResearchOutput


def run_eval_scenarios(
    arch_key: str,
    data_sources,
    manager: FoundryAgentManager,
    max_results: int
) -> Dict[str, Any]:
    """Run all BDD eval scenarios and return aggregated results."""
    arch_info = ARCHITECTURES[arch_key]
    scenarios = get_all_scenarios()
    # Import scenarios module to register them
    import evaluation.scenarios  # noqa: F401

    runner = EvalRunner(output_dir="eval_results")
    orchestrator = arch_info["class"](manager, data_sources)
    results = runner.run_all(orchestrator, arch_key, scenarios)

    successful = [r for r in results if not r.error]
    overall = sum(r.overall_score for r in successful) / len(successful) if successful else 0.0

    return {
        "mode": "eval",
        "scenario_count": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "success_rate": len(successful) / len(results) if results else 0,
        "overall_score": round(overall, 3),
        "total_time": sum(r.completion_time for r in results),
        "avg_time": sum(r.completion_time for r in results) / len(results) if results else 0,
        "scenarios": [
            {
                "id": r.scenario_id,
                "name": r.scenario_name,
                "score": round(r.overall_score, 3),
                "passed": r.passed,
                "time": round(r.completion_time, 2),
                "error": r.error,
            }
            for r in results
        ],
    }


def run_single_query(
    arch_key: str,
    query: str,
    data_sources,
    manager: FoundryAgentManager,
    max_results: int
) -> Dict[str, Any]:
    """Run a single query and return results."""
    arch_info = ARCHITECTURES[arch_key]
    orchestrator = arch_info["class"](manager, data_sources)
    result = orchestrator.research(query, max_results_per_source=max_results)

    return {
        "query": result.query,
        "answer": result.answer,
        "answer_length": len(result.answer),
        "documents_retrieved": result.documents_retrieved,
        "documents_used": result.documents_used,
        "citations": result.citations,
        "citation_count": len(result.citations),
        "time_elapsed": result.time_elapsed,
        "sources_checked": result.sources_checked,
        "metadata": result.metadata,
    }


def run_architecture(
    arch_key: str,
    output_dir: Path,
    query: Optional[str] = None,
    eval_mode: bool = False,
    max_results: int = 5,
    keep_agents: bool = False,
) -> dict:
    """Run architecture and manage status files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    in_progress_file = output_dir / f"{arch_key}_in_progress"
    complete_file = output_dir / f"{arch_key}_complete.json"
    failed_file = output_dir / f"{arch_key}_failed.json"
    
    # Clean up any previous run files
    for f in [in_progress_file, complete_file, failed_file]:
        if f.exists():
            f.unlink()
    
    mode = "eval" if eval_mode else "query"
    
    # Create in_progress marker
    in_progress_file.write_text(json.dumps({
        "architecture": arch_key,
        "started_at": datetime.now().isoformat(),
        "mode": mode,
        "query": query if query else "(eval fixtures)",
    }))
    
    arch_info = ARCHITECTURES[arch_key]
    print(f"🚀 Starting {arch_info['name']} ({mode} mode)...")
    
    try:
        data_sources = get_all_sources()
        manager = FoundryAgentManager(keep_agents=keep_agents)
        
        with manager:
            if eval_mode:
                result_data = run_eval_scenarios(arch_key, data_sources, manager, max_results)
            else:
                result_data = run_single_query(arch_key, query, data_sources, manager, max_results)
        
        output = {
            "architecture": arch_key,
            "name": arch_info["name"],
            "emoji": arch_info.get("emoji", ""),
            "success": True,
            "mode": mode,
            "completed_at": datetime.now().isoformat(),
            **result_data,
        }
        
        # Write complete file
        complete_file.write_text(json.dumps(output, indent=2, default=str))
        
        # Remove in_progress marker
        in_progress_file.unlink()
        
        if eval_mode:
            print(f"✅ {arch_info['name']} completed {result_data['successful']}/{result_data['scenario_count']} scenarios, score: {result_data['overall_score']:.2f}")
        else:
            print(f"✅ {arch_info['name']} completed in {result_data['time_elapsed']:.1f}s")
        
        return output
        
    except Exception as e:
        import traceback
        error_output = {
            "architecture": arch_key,
            "name": arch_info["name"],
            "success": False,
            "mode": mode,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "failed_at": datetime.now().isoformat(),
        }
        
        # Write failed file
        failed_file.write_text(json.dumps(error_output, indent=2))
        
        # Remove in_progress marker
        if in_progress_file.exists():
            in_progress_file.unlink()
        
        print(f"❌ {arch_info['name']} failed: {e}")
        return error_output


def main():
    parser = argparse.ArgumentParser(description="Run single architecture")
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        required=True,
        choices=list(ARCHITECTURES.keys()),
        help="Architecture to run"
    )
    
    # Mode: either --query or --eval
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--query", "-q",
        type=str,
        help="Research query (single query mode)"
    )
    mode_group.add_argument(
        "--eval", "-e",
        action="store_true",
        help="Run eval fixtures (eval mode)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="comparison_run",
        help="Output directory for status files"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max results per source"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG)"
    )
    parser.add_argument(
        "--keep-agents",
        action="store_true",
        help="Don't delete agents after run (inspect them in Foundry portal)"
    )
    
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    run_architecture(
        args.architecture,
        Path(args.output_dir),
        query=args.query,
        eval_mode=args.eval,
        max_results=args.max_results,
        keep_agents=args.keep_agents,
    )


if __name__ == "__main__":
    main()
