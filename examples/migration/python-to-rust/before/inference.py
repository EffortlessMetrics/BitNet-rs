"""Legacy pure-Python BitNet-style inference example.

This intentionally small example mirrors common Python migration pain points:
manual decode loops, dict-shaped results, dynamic runtime validation, and
benchmark timing code mixed into the inference path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LegacyBitNetModel:
    """Tiny stand-in for a Python-only model implementation."""

    model_path: str
    eos_token: int = 0

    def tokenize(self, prompt: str) -> list[int]:
        return [len(part) for part in prompt.split()]

    def forward(self, tokens: list[int]) -> list[list[float]]:
        # A real implementation would run Python/Numpy kernels here. The last
        # row is shaped so argmax selects a deterministic next token.
        next_token = (sum(tokens) % 31) + 1 if tokens else 1
        row = [0.0] * 32
        row[next_token] = 1.0
        return [row]

    def detokenize(self, tokens: list[int]) -> str:
        return " ".join(f"token_{token}" for token in tokens)


def load_model(model_path: str) -> LegacyBitNetModel:
    if not model_path:
        raise ValueError("model_path cannot be empty")
    return LegacyBitNetModel(model_path=model_path)


class BitNetInference:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def generate(self, prompt: str, max_tokens: int = 100) -> dict[str, Any]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        start_time = time.perf_counter()
        tokens = self.model.tokenize(prompt)
        output_tokens: list[int] = []

        for _ in range(max_tokens):
            logits = self.model.forward(tokens)
            next_token = max(range(len(logits[-1])), key=logits[-1].__getitem__)
            output_tokens.append(next_token)
            tokens.append(next_token)

            if next_token == self.model.eos_token:
                break

        inference_time = time.perf_counter() - start_time
        token_count = len(output_tokens)

        return {
            "text": self.model.detokenize(output_tokens),
            "tokens": token_count,
            "time": inference_time,
            "tokens_per_second": token_count / inference_time if inference_time else 0.0,
        }


if __name__ == "__main__":
    model = BitNetInference("model.gguf")
    result = model.generate("The future of AI is")
    print(f"Generated: {result['text']}")
    print(f"Speed: {result['tokens_per_second']:.1f} tok/s")
