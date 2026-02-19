"""
Main entry point for the General Researcher POC.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow absolute imports from src
sys.path.insert(0, str(Path(__file__).parent))

from utils.tracing import setup_tracing
from data_sources import get_all_sources
from agents.client import FoundryAgentManager
from architectures import ARCHITECTURES


def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize tracing before any LLM calls
    setup_tracing()
    
    parser = argparse.ArgumentParser(description="General Researcher POC")
    parser.add_argument(
        "--query",
        type=str,
        default="What legislation was introduced in 2024 related to artificial intelligence?",
        help="Research query"
    )
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        choices=list(ARCHITECTURES.keys()),
        default="single_agent",
        help="Architecture to use (default: single_agent)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per source"
    )
    parser.add_argument(
        "--govinfo-api-key",
        type=str,
        default=None,
        help="GovInfo API key (optional, defaults to GOVINFO_API_KEY env or DEMO_KEY)"
    )
    parser.add_argument(
        "--list-architectures",
        action="store_true",
        help="List available architectures and exit"
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
    
    args = parser.parse_args()
    
    # List architectures if requested
    if args.list_architectures:
        print("Available architectures:")
        for key, info in ARCHITECTURES.items():
            print(f"  {key:<20} - {info['name']}")
        return
    
    arch_info = ARCHITECTURES[args.architecture]
    
    print("=" * 80)
    print(f"GENERAL RESEARCHER POC - {arch_info['name']}")
    print("=" * 80)
    
    # Initialize components
    data_sources = get_all_sources(govinfo_api_key=args.govinfo_api_key)
    manager = FoundryAgentManager(govinfo_api_key=args.govinfo_api_key, keep_agents=args.keep_agents)
    
    # Create agent/orchestrator
    with manager:
        orchestrator = arch_info["class"](manager, data_sources)
        
        # Execute research
        result = orchestrator.research(args.query, max_results_per_source=args.max_results)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESEARCH RESULTS")
    print("=" * 80)
    print(f"\n📋 Query: {result.query}")
    print(f"\n⏱️  Time: {result.time_elapsed:.2f}s")
    print(f"📊 Documents Retrieved: {result.documents_retrieved}")
    print(f"📊 Documents Used: {result.documents_used}")
    print(f"🔍 Sources: {', '.join(result.sources_checked) if result.sources_checked else 'N/A'}")
    print(f"\n💡 Answer:\n{'-' * 80}")
    print(result.answer)
    print("\n" + "=" * 80)
    print(f"✓ Research complete with {len(result.citations)} citations")
    print("=" * 80)


if __name__ == "__main__":
    main()
