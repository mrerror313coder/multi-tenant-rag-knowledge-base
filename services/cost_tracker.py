"""Cost and token tracking service for LLM calls."""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Price per 1M tokens (USD)
PRICING_TABLE = {
    "gemini-2.0-flash": {"prompt": 0.10 / 1_000_000, "completion": 0.40 / 1_000_000},
    "gemini-1.5-flash": {"prompt": 0.075 / 1_000_000, "completion": 0.30 / 1_000_000},
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "llama-3.3-70b-versatile": {"prompt": 0.59 / 1_000_000, "completion": 0.79 / 1_000_000},
    "mock": {"prompt": 0.0, "completion": 0.0},
}


class CostTracker:
    """Calculates tokens, cost estimates, and latency for RAG requests."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from raw text."""
        if not text:
            return 0
        return max(1, int(len(text.split()) * 1.3))

    @classmethod
    def calculate_cost(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculates estimated USD cost based on token counts."""
        matched_model = "mock"
        for key in PRICING_TABLE:
            if key in model_name.lower():
                matched_model = key
                break

        rates = PRICING_TABLE.get(matched_model, {"prompt": 0.0, "completion": 0.0})
        cost = (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
        return round(cost, 6)

    @classmethod
    def log_call(
        cls,
        org_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        is_success: bool = True,
        error: str = None,
    ) -> Dict[str, Any]:
        """Logs the LLM invocation metrics."""
        cost = cls.calculate_cost(model, prompt_tokens, completion_tokens)
        record = {
            "org_id": org_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
            "latency_ms": round(latency_ms, 2),
            "success": is_success,
            "error": error,
        }
        logger.info(f"[CostTracker] Org: {org_id} | Model: {model} | Tokens: {record['total_tokens']} | Cost: ${cost:.6f} | Latency: {record['latency_ms']}ms")
        return record
