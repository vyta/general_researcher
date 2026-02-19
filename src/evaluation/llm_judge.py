"""
LLM-based judge for semantic eval assertions.

Uses the OpenAI Responses API (same client as the agent manager)
to evaluate whether an answer meets qualitative criteria like
"comprehensive", "well-structured", "accurately cites sources".
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a government research agent.
You will be given a research query, the agent's answer, and a quality criterion.
Evaluate whether the answer meets the criterion on a scale of 0.0 to 1.0.

Respond with JSON only:
{
  "score": 0.0-1.0,
  "passed": true/false,
  "reasoning": "1-2 sentence explanation"
}

Score guide:
  0.0 = completely fails the criterion
  0.3 = partially addresses it with major gaps
  0.5 = adequately addresses it with some gaps
  0.7 = mostly meets the criterion
  1.0 = fully meets the criterion"""


class LLMJudge:
    """Evaluates qualitative assertions using an LLM."""

    def __init__(self, openai_client, model: str = "gpt-4.1"):
        self.client = openai_client
        self.model = model

    def judge_quality(self, answer: str, query: str, quality: str) -> dict:
        """Judge whether an answer meets a quality descriptor (e.g. 'comprehensive')."""
        return self._judge(
            answer, query,
            f"Is this answer '{quality}'? Consider the research query and evaluate "
            f"whether the answer genuinely demonstrates this quality."
        )

    def judge_criteria(self, answer: str, query: str, criteria: str) -> dict:
        """Judge whether an answer meets a free-form criterion."""
        return self._judge(
            answer, query,
            f"Does this answer meet the following criterion: '{criteria}'?"
        )

    def _judge(self, answer: str, query: str, question: str) -> dict:
        """Run an LLM judgment and parse the result."""
        prompt = (
            f"Research query: {query}\n\n"
            f"Agent's answer (first 2000 chars):\n{answer[:2000]}\n\n"
            f"Evaluation question: {question}"
        )
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "developer", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.output_text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            logger.debug("LLM judge result: %s", result)
            score = float(result.get("score", 1.0 if result.get("passed") else 0.0))
            return {
                "score": max(0.0, min(1.0, score)),
                "passed": bool(result.get("passed", score >= 0.7)),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
            return {"score": 0.0, "passed": False, "reasoning": f"Judge error: {e}"}
