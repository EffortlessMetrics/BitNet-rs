#!/usr/bin/env python3
"""Compare legacy fast_orig implementation with generate module.

This script defines sample prompts and expected outputs, attempts to
invoke `fast_orig` and `generate` modules to produce tokens, measures
execution time, and evaluates accuracy against expected outputs.

Results are printed as a structured JSON object.
"""
import json
import time
from typing import Any, Dict, List

# Sample prompts and expected outputs from the legacy model
SAMPLES: List[Dict[str, str]] = [
    {"prompt": "Hello", "expected": "Hello world"},
    {
        "prompt": "The capital of France is",
        "expected": "The capital of France is Paris",
    },
]


def try_import(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # ImportError or others
        return None, str(exc)


def generate_with(module: Any, prompt: str) -> Dict[str, Any]:
    """Attempt to generate tokens using a module."""
    start = time.perf_counter()
    try:
        if hasattr(module, "generate_tokens"):
            tokens = module.generate_tokens(prompt)  # type: ignore[attr-defined]
        else:
            tokens = module.generate(prompt)  # type: ignore[attr-defined]
        duration = time.perf_counter() - start
        token_count = len(tokens) if isinstance(tokens, list) else len(str(tokens).split())
        tps = token_count / duration if duration > 0 else None
        return {"output": tokens, "time": duration, "tokens_per_second": tps}
    except Exception as exc:
        duration = time.perf_counter() - start
        return {"error": str(exc), "time": duration}


fast_orig_mod, fast_orig_err = try_import("fast_orig")
generate_mod, generate_err = try_import("generate")

results: Dict[str, Any] = {"samples": [], "summary": {}}
fast_times: List[float] = []
fast_correct = 0
gen_times: List[float] = []
gen_correct = 0

for sample in SAMPLES:
    entry: Dict[str, Any] = {
        "prompt": sample["prompt"],
        "expected": sample["expected"],
    }

    if fast_orig_mod:
        fast_res = generate_with(fast_orig_mod, sample["prompt"])
        if "output" in fast_res and fast_res["output"] == sample["expected"]:
            fast_correct += 1
        if "time" in fast_res:
            fast_times.append(fast_res["time"])
        entry["fast_orig"] = fast_res
    else:
        entry["fast_orig"] = {"error": fast_orig_err}

    if generate_mod:
        gen_res = generate_with(generate_mod, sample["prompt"])
        if "output" in gen_res and gen_res["output"] == sample["expected"]:
            gen_correct += 1
        if "time" in gen_res:
            gen_times.append(gen_res["time"])
        entry["generate"] = gen_res
    else:
        entry["generate"] = {"error": generate_err}

    results["samples"].append(entry)

if fast_times:
    results["summary"]["fast_orig"] = {
        "avg_time": sum(fast_times) / len(fast_times),
        "accuracy": fast_correct / len(SAMPLES),
    }
else:
    results["summary"]["fast_orig"] = {"error": fast_orig_err or "not available"}

if gen_times:
    results["summary"]["generate"] = {
        "avg_time": sum(gen_times) / len(gen_times),
        "accuracy": gen_correct / len(SAMPLES),
    }
else:
    results["summary"]["generate"] = {"error": generate_err or "not available"}

print(json.dumps(results, indent=2))
