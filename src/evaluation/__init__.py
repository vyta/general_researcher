"""BDD-style evaluation framework for the General Researcher."""
from evaluation.dsl import scenario, get_all_scenarios, get_scenarios_by_category
from evaluation.runner import EvalRunner
from evaluation.llm_judge import LLMJudge

__all__ = [
    "scenario",
    "get_all_scenarios",
    "get_scenarios_by_category",
    "EvalRunner",
    "LLMJudge",
]
