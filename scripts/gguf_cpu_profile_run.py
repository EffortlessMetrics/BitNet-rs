#!/usr/bin/env python3
"""Run explicit Lunar Lake Rust GGUF CPU heavy-profile timing cases.

This script fills the CPU side of the same profile gap covered by
openvino_genai_profile_run.py for OpenVINO. It drives the existing
`bitnet slm-warm-session` command, then normalizes the per-profile warm-session
receipts into the compact `intel_258v_dense_slm_cpu_profile_run` shape consumed
by `bitnet lunar-lake profile-compare --cpu-profile-run`.

The output is timing/profile evidence only. It is not a route-promotion
decision and it does not claim speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


ROUTE_POLICY_SENTENCE = (
    "Route policy evidence keeps CPU, GPU, and NPU claims separate. "
)

PROFILE_CASES: list[dict[str, Any]] = [
    {
        "id": "prefill_heavy_route_policy_long_context",
        "profile": "prefill_heavy",
        "category": "long_prompt_summarization",
        "question": (
            "Summarize the following route-policy note in one short sentence.\n\n"
            + ROUTE_POLICY_SENTENCE * 192
        ),
        "max_new_tokens": 64,
        "profile_requirements": {"prompt_tokens": ">=2048", "output_tokens": "<=64"},
        "gate": {"kind": "readable", "min_words": 8},
        "min_generated_tokens": 1,
        "min_distinct_generated_tokens": 1,
    },
    {
        "id": "decode_heavy_route_policy_long_generation",
        "profile": "decode_heavy",
        "category": "decode_heavy",
        "question": (
            "Write a detailed technical guide of at least 900 words about designing a "
            "local inference route policy for CPU, GPU, and NPU. Use plain paragraphs "
            "and continue until complete."
        ),
        "max_new_tokens": 512,
        "profile_requirements": {"prompt_tokens": "<=256", "output_tokens": ">=512"},
        "gate": {"kind": "readable", "min_words": 120},
        "min_generated_tokens": 512,
        "min_distinct_generated_tokens": 8,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="Qwen2.5 Q8_0 GGUF model path")
    parser.add_argument(
        "--bitnet-bin",
        default="target/debug/bitnet.exe" if sys.platform == "win32" else "target/debug/bitnet",
        type=Path,
        help="Compiled bitnet CLI binary used to run slm-warm-session",
    )
    parser.add_argument(
        "--from-warm-session",
        action="append",
        default=[],
        type=Path,
        help="Existing slm-warm-session aggregate receipt to normalize instead of running it",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["prefill_heavy", "decode_heavy"],
        choices=["prefill_heavy", "decode_heavy"],
    )
    parser.add_argument("--work-dir", type=Path, default=Path("target/lunar-lake-cpu-profile-run"))
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--created-utc")
    parser.add_argument("--timeout-seconds", default=3600, type=int)
    parser.add_argument("--prompt-template", default="qwen2.5")
    parser.add_argument("--model-sha256")
    parser.add_argument(
        "--allow-threshold-miss",
        action="store_true",
        help="Write the receipt even if a case does not satisfy its profile token thresholds",
    )
    return parser.parse_args()


def utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def selected_cases(profiles: list[str]) -> list[dict[str, Any]]:
    selected = [case for case in PROFILE_CASES if case["profile"] in set(profiles)]
    if not selected:
        raise SystemExit("no profile cases selected")
    return selected


def requirement_satisfied(value: int | None, requirement: str) -> bool | None:
    requirement = requirement.strip()
    if requirement.startswith(">="):
        return value is not None and value >= int(requirement[2:].strip())
    if requirement.startswith("<="):
        return value is not None and value <= int(requirement[2:].strip())
    return None


def yaml_block(value: str, indent: int = 6) -> str:
    prefix = " " * indent
    return "\n".join(prefix + line if line else prefix for line in value.splitlines())


def write_profile_corpus(path: Path, case: dict[str, Any], model: Path, args: argparse.Namespace) -> None:
    gate = case["gate"]
    text = f"""schema: 1
artifact_kind: slm_cpu_warm_session_corpus
name: lunar-lake-cpu-profile-run-{case['profile']}
description: Rust GGUF CPU heavy-profile timing case for Lunar Lake route comparison.
model:
  repo: Qwen/Qwen2.5-0.5B-Instruct-GGUF
  file: {model.name}
  sha256: {args.model_sha256 or ''}
  family: qwen
  architecture: qwen2
  quant_format: Q8_0
defaults:
  prompt_template: {args.prompt_template}
  max_new_tokens: {case['max_new_tokens']}
  greedy: true
  deterministic: true
  temperature: 0.0
  top_k: 0
  repeat_runs: 1
  min_generated_tokens: {case['min_generated_tokens']}
  min_distinct_generated_tokens: {case['min_distinct_generated_tokens']}
cases:
  - id: {case['id']}
    question: |
{yaml_block(case['question'])}
    min_generated_tokens: {case['min_generated_tokens']}
    min_distinct_generated_tokens: {case['min_distinct_generated_tokens']}
    gate:
      kind: {gate['kind']}
      min_words: {gate['min_words']}
"""
    write_text_lf(path, text)


def run_warm_session(case: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.model is None:
        raise SystemExit("--model is required when --from-warm-session is not used")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.work_dir / f"{case['id']}.yaml"
    receipt_path = args.work_dir / f"{case['id']}-warm-session.json"
    write_profile_corpus(corpus_path, case, args.model, args)

    cmd = [
        str(args.bitnet_bin),
        "slm-warm-session",
        "--device",
        "cpu",
        "--model",
        str(args.model),
        "--corpus",
        str(corpus_path),
        "--corpus-repeat-runs",
        "1",
        "--strict-tokenizer",
        "--strict-loader",
        "--greedy",
        "--deterministic",
        "--threads",
        str(args.threads),
        "--prompt-template",
        args.prompt_template,
        "--quiet",
        "--json-out",
        str(receipt_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_seconds,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"slm-warm-session failed for {case['profile']} with exit {completed.returncode}\n"
            f"command: {' '.join(cmd)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    sidecar = receipt_path.with_suffix(".command.json")
    write_text_lf(
        sidecar,
        json.dumps(
            {
                "command": cmd,
                "wall_ms": round(wall_ms, 3),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
            indent=2,
        )
        + "\n",
    )
    return receipt_path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_receipt_path(path_value: Any, base: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    candidate = base.parent / path
    if candidate.exists():
        return candidate
    return path


def nested(mapping: dict[str, Any], path: str) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def find_prompt_summary(receipt: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    prompts = receipt.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("warm-session receipt does not contain prompts[]")
    for prompt in prompts:
        if isinstance(prompt, dict) and prompt.get("case_id") == case["id"]:
            return prompt
    if len(prompts) == 1 and isinstance(prompts[0], dict):
        return prompts[0]
    raise ValueError(f"warm-session receipt does not contain case {case['id']}")


def normalize_warm_session(receipt_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    prompt = find_prompt_summary(receipt, case)
    prompt_receipt_path = resolve_receipt_path(prompt.get("receipt_path"), receipt_path)
    prompt_receipt = read_json(prompt_receipt_path) if prompt_receipt_path and prompt_receipt_path.exists() else {}
    timing = prompt.get("timing") if isinstance(prompt.get("timing"), dict) else {}

    prompt_tokens = (
        integer(nested(prompt_receipt, "tokens.prompt"))
        or integer(prompt.get("prompt_token_count"))
        or integer(nested(prompt, "tokens.prompt"))
    )
    generated_tokens = (
        integer(prompt.get("generated_tokens"))
        or integer(nested(prompt_receipt, "tokens.generated"))
        or integer(nested(prompt, "tokens.generated"))
    )
    prompt_req = case["profile_requirements"]["prompt_tokens"]
    output_req = case["profile_requirements"]["output_tokens"]
    prompt_match = requirement_satisfied(prompt_tokens, prompt_req)
    output_match = requirement_satisfied(generated_tokens, output_req)
    profile_satisfied = (prompt_match is not False) and (output_match is not False)

    session_timing = receipt.get("timing") if isinstance(receipt.get("timing"), dict) else {}
    model_load_ms = number(session_timing.get("model_load_ms"))
    tokenizer_load_ms = number(session_timing.get("tokenizer_load_ms"))
    tokenize_ms = number(timing.get("tokenize_ms"))
    prompt_total_ms = number(timing.get("total_ms"))
    total_response_parts = [
        value
        for value in [model_load_ms, tokenizer_load_ms, prompt_total_ms]
        if value is not None
    ]
    total_response_ms = sum(total_response_parts) if total_response_parts else None

    fallback_used = bool(
        prompt.get("backend", {}).get("fallback_used", receipt.get("fallback_used", True))
    )
    quality = prompt.get("quality") if isinstance(prompt.get("quality"), dict) else {}
    generated_text = prompt.get("text")
    model = receipt.get("model") if isinstance(receipt.get("model"), dict) else {}
    tokenizer = receipt.get("tokenizer") if isinstance(receipt.get("tokenizer"), dict) else {}
    kernel_id = nested(prompt_receipt, "kernel.kernel_id") or "dense-qwen-cpu-reference"

    return {
        "id": case["id"],
        "profile": case["profile"],
        "category": case["category"],
        "route_id": "dense_slm_default_cpu",
        "question_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
        "prompt_template": nested(receipt, "generation.prompt_template") or "qwen2.5",
        "max_new_tokens": case["max_new_tokens"],
        "profile_requirements": case["profile_requirements"],
        "profile_requirement_status": {
            "prompt_tokens_match": prompt_match,
            "output_tokens_match": output_match,
            "profile_satisfied": profile_satisfied,
        },
        "prompt_token_count": prompt_tokens,
        "generated_token_count": generated_tokens,
        "generated_text": generated_text,
        "decoded_preview": generated_text[:240] if isinstance(generated_text, str) else None,
        "generated_token_ids": prompt.get("generated_token_ids"),
        "generated_token_ids_available": isinstance(prompt.get("generated_token_ids"), list),
        "generated_token_ids_source": "slm_warm_session_generated_ids",
        "quality": quality,
        "answer_gate": {
            "kind": case["gate"]["kind"],
            "passed": bool(quality.get("passed", False)),
            "source": "slm_warm_session_quality_gate",
        },
        "timing": {
            "model_load_ms": model_load_ms,
            "tokenizer_load_ms": tokenizer_load_ms,
            "tokenize_ms": tokenize_ms,
            "prefill_ms": number(timing.get("prefill_ms")),
            "first_token_ms": number(timing.get("first_token_ms")),
            "time_to_first_token_ms": number(timing.get("time_to_first_token_ms")),
            "decode_total_ms": number(timing.get("decode_total_ms")),
            "generation_total_ms": prompt_total_ms,
            "generation_wall_ms": prompt_total_ms,
            "total_response_ms": total_response_ms,
            "decode_steady_state_tok_s": number(timing.get("decode_steady_state_tok_s")),
        },
        "model": {
            "path": model.get("path"),
            "file": model.get("file"),
            "sha256": model.get("sha256"),
            "family": model.get("family"),
            "architecture": model.get("architecture"),
            "format": model.get("format"),
            "quantization": "Q8_0",
        },
        "tokenizer": {
            "source": tokenizer.get("source"),
            "type": tokenizer.get("type"),
            "strict": tokenizer.get("strict"),
        },
        "source_warm_session_receipt": str(receipt_path),
        "source_prompt_receipt": str(prompt_receipt_path) if prompt_receipt_path else None,
        "fallback_used": fallback_used,
        "requested_backend": "cpu",
        "selected_backend": "cpu-rust",
        "backend_lane": "dense_slm_default_cpu",
        "runtime_api": "cpu",
        "selected_kernel_or_runtime": kernel_id,
    }


def case_for_receipt(path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    receipt = read_json(path)
    prompts = receipt.get("prompts")
    if isinstance(prompts, list):
        case_ids = {prompt.get("case_id") for prompt in prompts if isinstance(prompt, dict)}
        for case in cases:
            if case["id"] in case_ids:
                return case
    if len(cases) == 1:
        return cases[0]
    raise ValueError(f"cannot infer profile case for warm-session receipt {path}")


def main() -> int:
    args = parse_args()
    created_utc = args.created_utc or utc_now()
    cases = selected_cases(args.profiles)

    receipt_paths: list[Path]
    if args.from_warm_session:
        receipt_paths = args.from_warm_session
    else:
        receipt_paths = [run_warm_session(case, args) for case in cases]

    normalized_cases: list[dict[str, Any]] = []
    for receipt_path in receipt_paths:
        case = case_for_receipt(receipt_path, cases)
        normalized_cases.append(normalize_warm_session(receipt_path, case))

    cases_by_profile = {case["profile"]: case for case in normalized_cases}
    missing_profiles = [case["profile"] for case in cases if case["profile"] not in cases_by_profile]
    if missing_profiles:
        raise SystemExit(f"missing CPU profile-run cases: {', '.join(missing_profiles)}")

    threshold_misses = [
        case["profile"]
        for case in normalized_cases
        if not case["profile_requirement_status"]["profile_satisfied"]
    ]
    fallback_used_any = any(bool(case["fallback_used"]) for case in normalized_cases)
    if threshold_misses and not args.allow_threshold_miss:
        raise SystemExit(
            "CPU profile-run threshold miss for "
            + ", ".join(threshold_misses)
            + "; pass --allow-threshold-miss to write diagnostic output anyway"
        )
    if fallback_used_any:
        raise SystemExit("CPU profile-run requires fallback_used=false for every case")

    first_model = normalized_cases[0].get("model", {}) if normalized_cases else {}
    first_tokenizer = normalized_cases[0].get("tokenizer", {}) if normalized_cases else {}
    out = {
        "schema_version": "1.0.0",
        "artifact_kind": "intel_258v_dense_slm_cpu_profile_run",
        "campaign": "intel-258v-platform",
        "item": "LNL258V-PROFILE-RUN-004",
        "created_utc": created_utc,
        "machine_id": args.machine_id,
        "proof_stage": "rust_gguf_cpu_heavy_profile_timing_evidence",
        "purpose": (
            "Record explicit Rust GGUF CPU prefill_heavy and decode_heavy timing cases so "
            "route-profile comparison can benchmark OpenVINO heavy-profile candidates against "
            "a profile-specific CPU baseline."
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "model": {
            "model_path": str(args.model) if args.model else first_model.get("path"),
            "model_family": first_model.get("family") or "qwen",
            "model_architecture": first_model.get("architecture") or "qwen2",
            "model_name": "qwen2.5-0.5b-instruct",
            "format": first_model.get("format") or "gguf",
            "quantization": first_model.get("quantization") or "Q8_0",
            "sha256": first_model.get("sha256") or args.model_sha256,
        },
        "tokenizer": first_tokenizer,
        "profile_cases": [
            {
                "id": case["id"],
                "profile": case["profile"],
                "profile_requirements": case["profile_requirements"],
                "max_new_tokens": case["max_new_tokens"],
            }
            for case in PROFILE_CASES
            if case["profile"] in set(args.profiles)
        ],
        "cases": normalized_cases,
        "fallback_used": fallback_used_any,
        "verification": {
            "slm_warm_session_ran_for_all_profile_cases": len(normalized_cases) == len(cases),
            "all_profile_requirements_satisfied": not threshold_misses,
            "threshold_misses": threshold_misses,
            "fallback_used": fallback_used_any,
            "direct_generated_token_ids_available": all(
                bool(case.get("generated_token_ids_available")) for case in normalized_cases
            ),
            "route_promotion_changed": False,
            "profile_run_is_timing_evidence_not_quality_promotion": True,
        },
        "claim_boundary": {
            "may_claim": [
                "Rust GGUF CPU has explicit same-machine timing evidence for selected dense Qwen profile token thresholds when profile_requirement_status passes.",
                "The receipt records fallback_used=false, CPU backend/runtime identity, prompt token counts, generated token counts, generated token IDs, and warm-session timing for the requested CPU route.",
            ],
            "must_not_claim": [
                "Any route is newly promoted by this receipt.",
                "OpenVINO GPU or NPU speedup, acceleration, or power advantage is proven by this receipt alone.",
                "OpenVINO GPU evidence proves native OpenCL execution.",
                "OpenVINO NPU evidence proves native NPU inference outside OpenVINO GenAI.",
                "Dense SLM profile-run receipts prove BitNet QK256/I2_S behavior.",
                "BitNet QK256/I2_S behavior changed.",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(args.json_out, json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
