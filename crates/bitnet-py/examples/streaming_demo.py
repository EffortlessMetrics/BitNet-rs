#!/usr/bin/env python3
"""Minimal demonstration of live token streaming."""

import asyncio
import sys
from typing import AsyncIterator

try:
    import bitnet_py as bitnet
except ImportError:
    bitnet = None


async def stream_tokens(engine, prompt: str) -> AsyncIterator[str]:
    for token in engine.generate_stream(prompt):
        yield token


class MockEngine:
    def generate_stream(self, prompt: str):
        for token in ["Hello", " ", "from", " ", "BitNet"]:
            yield token


async def main() -> int:
    if bitnet and len(sys.argv) > 1:
        model_path = sys.argv[1]
        tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else "./tokenizer.model"
        model = bitnet.load_model(model_path, device="cpu")
        tokenizer = bitnet.create_tokenizer(tokenizer_path)
        engine = bitnet.SimpleInference(model, tokenizer)
        prompt = "Streaming demo"
    else:
        engine = MockEngine()
        prompt = "mock prompt"

    print(f"Prompt: {prompt}")
    print("Response:", end=" ", flush=True)
    async for token in stream_tokens(engine, prompt):
        print(token, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
