"""
Azure AI Evaluation SDK integration.

Wraps built-in evaluators (RelevanceEvaluator, CoherenceEvaluator,
GroundednessEvaluator, FluencyEvaluator) for use as BDD step types.

Evaluators return scores on a 1–5 integer scale. We normalize to 0.0–1.0.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class AzureEvaluators:
    """Lazy wrapper around azure.ai.evaluation built-in evaluators."""

    def __init__(self, model_config: dict):
        self._model_config = model_config
        self._relevance = None
        self._coherence = None
        self._groundedness = None
        self._fluency = None

    @classmethod
    def from_env(cls, credential=None) -> "AzureEvaluators":
        """Build model_config from environment variables and credential."""
        from azure.identity import DefaultAzureCredential
        import re

        endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
        # Extract resource name: https://{name}.services.ai.azure.com/...
        m = re.match(r"https://([^.]+)\.", endpoint)
        if not m:
            raise ValueError(f"Cannot extract resource name from AZURE_AI_PROJECT_ENDPOINT: {endpoint}")
        resource_name = m.group(1)

        cred = credential or DefaultAzureCredential()
        # Get a token and use it as api_key — the SDK's TypedDict validator
        # does strict type checking and credential objects can fail validation.
        token = cred.get_token("https://cognitiveservices.azure.com/.default").token

        model_config = {
            "azure_endpoint": f"https://{resource_name}.openai.azure.com/",
            "azure_deployment": os.environ.get("MODEL_DEPLOYMENT_NAME_FAST",
                                               os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o")),
            "api_key": token,
        }
        return cls(model_config)

    def _get_evaluator(self, name: str):
        """Lazily import and cache an evaluator instance."""
        attr = f"_{name}"
        if getattr(self, attr) is None:
            from azure.ai.evaluation import (
                RelevanceEvaluator,
                CoherenceEvaluator,
                GroundednessEvaluator,
                FluencyEvaluator,
            )
            evaluator_cls = {
                "relevance": RelevanceEvaluator,
                "coherence": CoherenceEvaluator,
                "groundedness": GroundednessEvaluator,
                "fluency": FluencyEvaluator,
            }[name]
            setattr(self, attr, evaluator_cls(model_config=self._model_config))
        return getattr(self, attr)

    def _normalize(self, score: int) -> float:
        """Normalize 1–5 integer score to 0.0–1.0."""
        return max(0.0, min(1.0, (score - 1) / 4))

    def evaluate_relevance(self, query: str, response: str) -> dict:
        """Evaluate how relevant the response is to the query."""
        try:
            evaluator = self._get_evaluator("relevance")
            result = evaluator(query=query, response=response)
            raw = result.get("relevance", 3)
            return {"score": self._normalize(raw), "raw": raw, "detail": f"relevance={raw}/5"}
        except Exception as e:
            logger.warning("Azure relevance evaluator failed: %s", e)
            return {"score": 0.0, "raw": 0, "detail": f"error: {e}"}

    def evaluate_coherence(self, query: str, response: str) -> dict:
        """Evaluate the logical flow and coherence of the response."""
        try:
            evaluator = self._get_evaluator("coherence")
            result = evaluator(query=query, response=response)
            raw = result.get("coherence", 3)
            return {"score": self._normalize(raw), "raw": raw, "detail": f"coherence={raw}/5"}
        except Exception as e:
            logger.warning("Azure coherence evaluator failed: %s", e)
            return {"score": 0.0, "raw": 0, "detail": f"error: {e}"}

    def evaluate_groundedness(self, query: str, response: str, context: str) -> dict:
        """Evaluate how well the response is grounded in the provided context."""
        try:
            evaluator = self._get_evaluator("groundedness")
            result = evaluator(query=query, response=response, context=context)
            raw = result.get("groundedness", 3)
            return {"score": self._normalize(raw), "raw": raw, "detail": f"groundedness={raw}/5"}
        except Exception as e:
            logger.warning("Azure groundedness evaluator failed: %s", e)
            return {"score": 0.0, "raw": 0, "detail": f"error: {e}"}

    def evaluate_fluency(self, response: str) -> dict:
        """Evaluate the fluency (grammar, vocabulary, naturalness) of the response."""
        try:
            evaluator = self._get_evaluator("fluency")
            result = evaluator(response=response)
            raw = result.get("fluency", 3)
            return {"score": self._normalize(raw), "raw": raw, "detail": f"fluency={raw}/5"}
        except Exception as e:
            logger.warning("Azure fluency evaluator failed: %s", e)
            return {"score": 0.0, "raw": 0, "detail": f"error: {e}"}
