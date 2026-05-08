"""Benchmark harness for the legacy Python example."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

from inference import BitNetInference


PROMPTS = (
    "The future of AI is",
    "Rust programming language",
    "Machine learning models",
    "High performance computing",
)


def benchmark_inference(
    model_path: str,
    prompts: Sequence[str] = PROMPTS,
    runs: int = 5,
) -> dict[str, float]:
    model = BitNetInference(model_path)
    times: list[float] = []
    token_counts: list[int] = []

    for _ in range(runs):
        for prompt in prompts:
            start = time.perf_counter()
            result = model.generate(prompt, max_tokens=50)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            token_counts.append(int(result["tokens"]))

    total_time = sum(times)
    total_tokens = sum(token_counts)
    return {
        "avg_time": statistics.mean(times),
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "total_tokens": float(total_tokens),
    }


if __name__ == "__main__":
    results = benchmark_inference("model.gguf")
    print(f"Legacy Python: {results['tokens_per_second']:.1f} tok/s")
