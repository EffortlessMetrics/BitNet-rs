#!/usr/bin/env python3
"""Run explicit Lunar Lake OpenVINO heavy-profile timing cases.

This is timing/profile evidence, not a route-promotion decision. The cases are
intentionally larger than answer-corpus-v2 so route-profile comparison can stop
borrowing tiny corpus cases for prefill-heavy and decode-heavy profile checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from openvino_genai_corpus_v2 import DEVICE_BACKENDS
from openvino_genai_corpus_v2 import evaluate_gate
from openvino_genai_corpus_v2 import file_record
from openvino_genai_corpus_v2 import normalize_text
from openvino_genai_corpus_v2 import perf_metrics
from openvino_genai_corpus_v2 import quality_status
from openvino_genai_corpus_v2 import summarize_cases
from openvino_genai_corpus_v2 import utc_now
from openvino_genai_token_utils import generate_with_direct_token_ids


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
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--devices", nargs="+", default=["GPU.0", "NPU"])
    parser.add_argument("--created-utc")
    parser.add_argument(
        "--npu-max-prompt-len",
        default=4096,
        type=int,
        help="MAX_PROMPT_LEN passed to the NPU LLMPipeline so prefill_heavy can exceed the NPU default 1024-token stateful limit.",
    )
    parser.add_argument(
        "--manifest",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-ir-qwen25-int4-sym-manifest.json",
    )
    parser.add_argument(
        "--route-profile-comparison",
        default="ci/hardware/intel-258v/2026-05-08/lunar-lake-route-profile-comparison.json",
    )
    return parser.parse_args()


def requirement_satisfied(value: int | None, requirement: str) -> bool | None:
    requirement = requirement.strip()
    if requirement.startswith(">="):
        return value is not None and value >= int(requirement[2:].strip())
    if requirement.startswith("<="):
        return value is not None and value <= int(requirement[2:].strip())
    return None


def run_device(
    device: str,
    model_dir: Path,
    ov_genai: Any,
    ov_core: Any,
    npu_max_prompt_len: int,
) -> dict[str, Any]:
    selected_backend, backend_lane, selected_runtime = DEVICE_BACKENDS.get(
        device,
        (
            f"openvino-{device.lower()}",
            f"dense_slm_openvino_{device.lower()}",
            f"openvino-genai-llmpipeline-{device.lower()}",
        ),
    )
    try:
        resolved_device = ov_core.get_property(device, "FULL_DEVICE_NAME")
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        resolved_device = f"unavailable: {type(exc).__name__}: {exc}"

    construct_start = time.perf_counter()
    pipeline_config: dict[str, Any] = {}
    if device.upper() == "NPU":
        pipeline_config["MAX_PROMPT_LEN"] = npu_max_prompt_len
    if pipeline_config:
        pipe = ov_genai.LLMPipeline(str(model_dir), device, pipeline_config)
    else:
        pipe = ov_genai.LLMPipeline(str(model_dir), device)
    construct_wall_ms = (time.perf_counter() - construct_start) * 1000.0
    tokenizer = pipe.get_tokenizer()

    results: list[dict[str, Any]] = []
    for case in PROFILE_CASES:
        chunks: list[dict[str, Any]] = []
        first_chunk_at: list[float | None] = [None]
        generation_start = time.perf_counter()

        def streamer(text: str) -> Any:
            now = time.perf_counter()
            if text and first_chunk_at[0] is None:
                first_chunk_at[0] = now
            chunks.append({"elapsed_ms": (now - generation_start) * 1000.0, "text": text})
            return ov_genai.StreamingStatus.RUNNING

        generation = generate_with_direct_token_ids(
            pipe,
            tokenizer,
            ov_genai,
            case["question"],
            int(case["max_new_tokens"]),
            streamer=streamer,
        )
        generation_wall_ms = (time.perf_counter() - generation_start) * 1000.0
        result = generation["result"]
        prompt = generation["prompt"]
        generated_text = generation["generated_text"]
        gate = evaluate_gate(case.get("gate"), generated_text)
        quality = quality_status(gate, {"kind": "none", "passed": True}, generated_text)
        first_chunk_ms = None
        if first_chunk_at[0] is not None:
            first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
        prompt_req = case["profile_requirements"]["prompt_tokens"]
        output_req = case["profile_requirements"]["output_tokens"]
        prompt_match = requirement_satisfied(prompt["prompt_token_count"], prompt_req)
        output_match = requirement_satisfied(generation["generated_token_count"], output_req)
        profile_satisfied = (prompt_match is not False) and (output_match is not False)
        results.append(
            {
                "id": case["id"],
                "profile": case["profile"],
                "category": case["category"],
                "question_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
                "prompt_template": "qwen2.5",
                "prompt": {"template_family": "qwen2.5", **prompt},
                "prompt_token_count": prompt["prompt_token_count"],
                "max_new_tokens": case["max_new_tokens"],
                "profile_requirements": case["profile_requirements"],
                "profile_requirement_status": {
                    "prompt_tokens_match": prompt_match,
                    "output_tokens_match": output_match,
                    "profile_satisfied": profile_satisfied,
                },
                "generation_config": {
                    "do_sample": False,
                    "num_beams": 1,
                    "apply_chat_template": True,
                    "max_new_tokens": case["max_new_tokens"],
                    "npu_constraints": {
                        "greedy_or_simple_generation": True,
                        "beam_search": False,
                        "parallel_sampling": False,
                    },
                },
                "generated_text": generated_text,
                "decoded_preview": generated_text[:240],
                "normalized_output": normalize_text(generated_text),
                "generated_token_ids": generation["generated_token_ids"],
                "generated_token_ids_available_from_pipeline": generation[
                    "generated_token_ids_available_from_pipeline"
                ],
                "generated_token_ids_source": generation["generated_token_ids_source"],
                "generated_token_count": generation["generated_token_count"],
                "quality": quality,
                "answer_gate": gate,
                "timing": {
                    "pipeline_construct_wall_ms": construct_wall_ms,
                    "generation_wall_ms": generation_wall_ms,
                    "first_streamed_text_chunk_ms": first_chunk_ms,
                    "first_streamed_text_chunk": chunks[0]["text"] if chunks else None,
                    "streamed_chunks_count": len(chunks),
                    "openvino_perf_metrics": perf_metrics(result),
                },
                "fallback_used": False,
                "requested_backend": selected_backend,
                "selected_backend": selected_backend,
                "backend_lane": backend_lane,
                "runtime_api": "openvino_genai",
                "runtime_device": device,
                "selected_kernel_or_runtime": selected_runtime,
            }
        )

    return {
        "runtime_device": device,
        "resolved_device_name": resolved_device,
        "requested_backend": selected_backend,
        "selected_backend": selected_backend,
        "backend_lane": backend_lane,
        "runtime_api": "openvino_genai",
        "selected_kernel_or_runtime": selected_runtime,
        "pipeline_construct_wall_ms": construct_wall_ms,
        "pipeline_config": pipeline_config,
        "fallback_used": False,
        "cases": results,
        "quality_summary": summarize_cases(results),
    }


def main() -> int:
    args = parse_args()
    created_utc = args.created_utc or utc_now()

    import openvino as ov
    import openvino_genai as ov_genai

    core = ov.Core()
    devices: list[dict[str, Any]] = []
    fallback_used_any = False
    for device in args.devices:
        record = run_device(device, args.model_dir, ov_genai, core, args.npu_max_prompt_len)
        devices.append(record)
        fallback_used_any = fallback_used_any or bool(record["fallback_used"])

    out = {
        "schema_version": "1.0.0",
        "artifact_kind": "intel_258v_dense_slm_openvino_profile_run",
        "campaign": "intel-258v-platform",
        "item": "LNL258V-PROFILE-RUN-001",
        "created_utc": created_utc,
        "machine_id": args.machine_id,
        "proof_stage": "openvino_heavy_profile_timing_evidence",
        "purpose": (
            "Record explicit OpenVINO prefill_heavy and decode_heavy timing cases so "
            "route-profile comparison no longer has to borrow tiny corpus-v2 cases for "
            "OpenVINO candidate route timing applicability."
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "model": {
            "model_dir": args.model_dir.as_posix(),
            "manifest": file_record(Path(args.manifest)),
            "model_family": "qwen",
            "model_architecture": "qwen2",
            "model_name": "qwen2.5-0.5b-instruct",
            "format": "openvino_ir",
            "quantization": "int4_sym",
        },
        "route_profile_comparison": args.route_profile_comparison,
        "profile_cases": [
            {
                "id": case["id"],
                "profile": case["profile"],
                "profile_requirements": case["profile_requirements"],
                "max_new_tokens": case["max_new_tokens"],
            }
            for case in PROFILE_CASES
        ],
        "generation": {
            "devices": devices,
            "fallback_used": fallback_used_any,
        },
        "verification": {
            "llmpipeline_constructed_for_all_devices": len(devices) == len(args.devices),
            "generation_ran_for_all_profile_cases": True,
            "fallback_used": fallback_used_any,
            "direct_generated_token_ids_available": True,
            "route_promotion_changed": False,
            "candidate_routes_remain_unpromoted": True,
            "profile_run_is_timing_evidence_not_quality_promotion": True,
        },
        "claim_boundary": {
            "may_claim": [
                "OpenVINO candidate routes have explicit same-machine timing evidence for prefill_heavy and decode_heavy profile token thresholds when the case profile_requirement_status passes.",
                "The receipt records fallback_used=false, runtime identity, prompt token counts, generated token counts, direct generated token IDs, and OpenVINO PerfMetrics for requested OpenVINO routes.",
            ],
            "must_not_claim": [
                "OpenVINO GPU or NPU is newly promoted by this receipt.",
                "OpenVINO GPU or NPU speedup, acceleration, or power advantage is proven by this receipt alone.",
                "OpenVINO GPU evidence proves native OpenCL execution.",
                "OpenVINO NPU evidence proves native NPU inference outside OpenVINO GenAI.",
                "Dense SLM profile-run receipts prove BitNet QK256/I2_S behavior.",
                "BitNet QK256/I2_S behavior changed.",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if not fallback_used_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
