"""
Cost tracking utilities for LLM API usage in synthetic data generation.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class CostEstimate:
    """Structure for cost estimation results."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    currency: str = "USD"

class CostTracker:
    """Track and estimate costs for different LLM providers."""
    
    # Updated pricing per 1M tokens (July 2025)
    PRICING = {
        "openai": {
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4.5-preview": {"input": 75.00, "output": 150.00}
        },
        "together": {
            "meta-llama/Llama-2-7b-chat-hf": {"input": 0.20, "output": 0.20},
            "meta-llama/Llama-2-13b-chat-hf": {"input": 0.30, "output": 0.30},
            "meta-llama/Llama-2-70b-chat-hf": {"input": 0.90, "output": 0.90},
            "meta-llama/Meta-Llama-3-8B-Instruct": {"input": 0.20, "output": 0.20},
            "meta-llama/Meta-Llama-3-70B-Instruct": {"input": 0.90, "output": 0.90}
        }
    }
    
    def __init__(self):
        self.session_costs = []
        self.total_cost = 0.0
    
    def estimate_cost(self, provider: str, model: str, input_tokens: int, 
                     output_tokens: int) -> CostEstimate:
        """Estimate cost for a given token usage."""
        
        # Normalize model name
        model_key = self._normalize_model_name(model)
        
        # Get pricing
        if provider in self.PRICING and model_key in self.PRICING[provider]:
            pricing = self.PRICING[provider][model_key]
        else:
            # Fallback to GPT-4o-mini pricing if unknown
            pricing = self.PRICING["openai"]["gpt-4o-mini"]
        
        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        estimate = CostEstimate(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost=total_cost
        )
        
        # Track in session
        self.session_costs.append(estimate)
        self.total_cost += total_cost
        
        return estimate
    
    def _normalize_model_name(self, model: str) -> str:
        """Normalize model names to match pricing keys."""
        model_lower = model.lower()
        
        # OpenAI model mapping
        if "gpt-4o-mini" in model_lower:
            return "gpt-4o-mini"
        elif "gpt-4.1-nano" in model_lower:
            return "gpt-4.1-nano"
        elif "gpt-4.1-mini" in model_lower:
            return "gpt-4.1-mini"
        elif "gpt-4.1" in model_lower:
            return "gpt-4.1"
        elif "gpt-4o" in model_lower:
            return "gpt-4o"
        elif "gpt-4.5" in model_lower:
            return "gpt-4.5-preview"
        
        # Together.ai Llama model mapping
        elif "meta-llama/llama-2-7b" in model_lower:
            return "meta-llama/Llama-2-7b-chat-hf"
        elif "meta-llama/llama-2-13b" in model_lower:
            return "meta-llama/Llama-2-13b-chat-hf"
        elif "meta-llama/llama-2-70b" in model_lower:
            return "meta-llama/Llama-2-70b-chat-hf"
        elif "meta-llama/meta-llama-3-8b" in model_lower:
            return "meta-llama/Meta-Llama-3-8B-Instruct"
        elif "meta-llama/meta-llama-3-70b" in model_lower:
            return "meta-llama/Meta-Llama-3-70B-Instruct"
        
        # Return original if no match
        return model
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of costs for current session."""
        if not self.session_costs:
            return {"total_cost": 0, "total_tokens": 0, "calls": 0}
        
        total_tokens = sum(cost.total_tokens for cost in self.session_costs)
        total_calls = len(self.session_costs)
        
        return {
            "total_cost": round(self.total_cost, 4),
            "total_tokens": total_tokens,
            "calls": total_calls,
            "average_cost_per_call": round(self.total_cost / total_calls, 4),
            "cost_per_1k_tokens": round((self.total_cost / total_tokens) * 1000, 4) if total_tokens > 0 else 0
        }
    
    def save_cost_report(self, filepath: str):
        """Save detailed cost report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_summary": self.get_session_summary(),
            "detailed_costs": [
                {
                    "provider": cost.provider,
                    "model": cost.model,
                    "input_tokens": cost.input_tokens,
                    "output_tokens": cost.output_tokens,
                    "total_tokens": cost.total_tokens,
                    "estimated_cost": cost.estimated_cost
                }
                for cost in self.session_costs
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_cost_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_session_summary()
        
        print("\n💰 Cost Summary")
        print("=" * 40)
        print(f"Total API Calls: {summary['calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"Average per Call: ${summary['average_cost_per_call']:.4f}")
        print(f"Cost per 1K Tokens: ${summary['cost_per_1k_tokens']:.4f}")
        
        if self.session_costs:
            print(f"\nProvider Breakdown:")
            providers = {}
            for cost in self.session_costs:
                if cost.provider not in providers:
                    providers[cost.provider] = {"cost": 0, "tokens": 0, "calls": 0}
                providers[cost.provider]["cost"] += cost.estimated_cost
                providers[cost.provider]["tokens"] += cost.total_tokens
                providers[cost.provider]["calls"] += 1
            
            for provider, data in providers.items():
                print(f"  {provider}: ${data['cost']:.4f} ({data['calls']} calls, {data['tokens']:,} tokens)")

def estimate_article_costs(num_articles: int, provider: str = "openai", 
                          model: str = "gpt-4o-mini") -> Dict[str, float]:
    """Estimate costs for processing a given number of articles."""
    
    tracker = CostTracker()
    
    # Estimate tokens per article
    input_tokens_per_article = 1200  # Article + prompts + schema
    output_tokens_per_article = 800   # Facts + synthetic article
    
    total_input = num_articles * input_tokens_per_article
    total_output = num_articles * output_tokens_per_article
    
    estimate = tracker.estimate_cost(provider, model, total_input, total_output)
    
    return {
        "articles": num_articles,
        "provider": provider,
        "model": model,
        "total_tokens": estimate.total_tokens,
        "estimated_cost": estimate.estimated_cost,
        "cost_per_article": estimate.estimated_cost / num_articles
    }

if __name__ == "__main__":
    # Example usage and cost comparison
    print("🔍 Cost Estimation for COVID Synthetic Data Generation")
    print("=" * 60)
    
    test_scales = [10, 100, 1000]
    test_providers = [
        ("together", "meta-llama/Meta-Llama-3-8B-Instruct"),  # Together.ai - cheapest
        ("openai", "gpt-4.1-nano"),   # OpenAI nano
        ("openai", "gpt-4o-mini"),    # OpenAI mini
        ("together", "meta-llama/Meta-Llama-3-70B-Instruct")  # Together.ai larger model
    ]
    
    for num_articles in test_scales:
        print(f"\n📊 Cost for {num_articles} articles:")
        for provider, model in test_providers:
            costs = estimate_article_costs(num_articles, provider, model)
            print(f"  {provider:12} {model:20} ${costs['estimated_cost']:6.2f} (${costs['cost_per_article']:6.4f}/article)")
