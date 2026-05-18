#!/usr/bin/env python3
"""Run a bounded Lunar Lake OpenVINO GenAI operator ask receipt.

This helper executes only explicit dense-SLM OpenVINO candidate routes from an
existing Lunar Lake operator-readiness receipt. It does not make an acceleration
claim; it records what OpenVINO GenAI ran and preserves strict fallback and claim
boundaries for the requested device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openvino_genai_token_utils import generate_with_direct_token_ids
from openvino_genai_token_utils import prompt_evidence as ov_prompt_evidence
from openvino_genai_token_utils import public_prompt_evidence


DEVICE_ROUTES = {
    "GPU.0": {
        "route_id": "dense_slm_openvino_gpu_candidate",
        "selected_backend": "openvino-gpu",
        "backend_lane": "dense_slm_openvino_gpu_arc140v",
        "selected_kernel_or_runtime": "openvino-genai-llmpipeline-gpu0",
    },
    "NPU": {
        "route_id": "dense_slm_openvino_npu_candidate",
        "selected_backend": "openvino-npu",
        "backend_lane": "dense_slm_openvino_npu",
        "selected_kernel_or_runtime": "openvino-genai-llmpipeline-npu",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=sorted(DEVICE_ROUTES))
    parser.add_argument("--question", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--artifact-root", default="ci/hardware/intel-258v/2026-05-08", type=Path)
    parser.add_argument("--operator-receipt", default="lunar-lake-operator-readiness.json", type=Path)
    parser.add_argument("--route-id")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--expect-contains")
    parser.add_argument("--created-utc")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def resolve_receipt(root: Path, receipt: Path) -> Path:
    if receipt.is_absolute() or receipt.exists():
        return receipt
    return root / receipt


def evidence_for_file(operator: dict[str, Any], file_name: str) -> dict[str, Any] | None:
    suffix = "/" + file_name
    for item in operator.get("evidence", []):
        path = str(item.get("path", "")).replace("\\", "/")
        if path == file_name or path.endswith(suffix) or path.endswith(file_name):
            return item
    return None


def route_from_operator(operator: dict[str, Any], route_id: str) -> dict[str, Any]:
    for route in operator.get("routes", []):
        if route.get("route_id") == route_id:
            return route
    raise SystemExit(f"route `{route_id}` is not present in the operator readiness receipt")


def validate_route(operator: dict[str, Any], route: dict[str, Any], expected: dict[str, str]) -> list[str]:
    failures: list[str] = []
    if not operator.get("operator_ready"):
        failures.append("operator_ready_false")
    if route.get("selected_backend") != expected["selected_backend"]:
        failures.append("route_selected_backend_mismatch")
    if route.get("runtime_api") != "openvino_genai":
        failures.append("route_runtime_api_not_openvino_genai")
    if route.get("acceleration_claim") is not False:
        failures.append("route_acceleration_claim_not_false")
    if route.get("fallback_policy") != "strict_no_fallback":
        failures.append("route_fallback_policy_not_strict")

    for key in ("answer_gate_evidence", "phase_evidence"):
        file_name = route.get(key)
        if not file_name:
            failures.append(f"route_{key}_missing")
            continue
        evidence = evidence_for_file(operator, str(file_name))
        if evidence is None:
            failures.append(f"route_{key}_not_indexed")
            continue
        if evidence.get("fallback_used") is not False:
            failures.append(f"route_{key}_fallback_not_false")
        if evidence.get("issues"):
            failures.append(f"route_{key}_issues")

    return failures


def mean_std(pair: Any) -> dict[str, float | None]:
    if pair is None:
        return {"mean_ms": None, "std_ms": None}
    return {
        "mean_ms": float(getattr(pair, "mean", 0.0)),
        "std_ms": float(getattr(pair, "std", 0.0)),
    }


def perf_metrics(result: Any) -> dict[str, Any]:
    perf = result.perf_metrics
    return {
        "load_time_ms": float(perf.get_load_time()),
        "tokenization": mean_std(perf.get_tokenization_duration()),
        "generate": mean_std(perf.get_generate_duration()),
        "inference": mean_std(perf.get_inference_duration()),
        "time_to_first_token": mean_std(perf.get_ttft()),
        "time_per_output_token": mean_std(perf.get_tpot()),
        "inter_token_latency": mean_std(perf.get_ipot()),
        "throughput": mean_std(perf.get_throughput()),
        "detokenization": mean_std(perf.get_detokenization_duration()),
        "num_input_tokens": int(perf.get_num_input_tokens()),
        "num_generated_tokens": int(perf.get_num_generated_tokens()),
    }


def normalize_answer(text: str) -> str:
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def prompt_evidence(tokenizer: Any, question: str) -> dict[str, Any]:
    return public_prompt_evidence(ov_prompt_evidence(tokenizer, question))


def answer_gate(normalized: str, expected: str | None) -> dict[str, Any]:
    if expected is not None:
        return {
            "kind": "contains",
            "expected": expected,
            "passed": expected in normalized,
            "failed_rules": [] if expected in normalized else ["expected_text_missing"],
        }
    return {
        "kind": "non_empty",
        "expected": None,
        "passed": bool(normalized),
        "failed_rules": [] if normalized else ["empty_answer"],
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        raise SystemExit("--max-new-tokens must be in 1..=128")

    expected = DEVICE_ROUTES[args.device]
    route_id = args.route_id or expected["route_id"]
    if route_id != expected["route_id"]:
        raise SystemExit(f"route `{route_id}` does not match device `{args.device}`")

    operator_path = resolve_receipt(args.artifact_root, args.operator_receipt)
    operator = read_json(operator_path)
    route = route_from_operator(operator, route_id)
    validation_failures = validate_route(operator, route, expected)
    if validation_failures:
        raise SystemExit("operator route validation failed: " + ", ".join(validation_failures))

    import openvino as ov
    import openvino_genai as ov_genai

    core = ov.Core()
    try:
        resolved_device = core.get_property(args.device, "FULL_DEVICE_NAME")
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        resolved_device = f"unavailable: {type(exc).__name__}: {exc}"

    construct_start = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(args.model_dir), args.device)
    pipeline_construct_wall_ms = (time.perf_counter() - construct_start) * 1000.0
    tokenizer = pipe.get_tokenizer()
    prompt = prompt_evidence(tokenizer, args.question)

    chunks: list[dict[str, Any]] = []
    generation_start = time.perf_counter()
    first_chunk_at: list[float | None] = [None]

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
        args.question,
        args.max_new_tokens,
        streamer=streamer,
    )
    generation_wall_ms = (time.perf_counter() - generation_start) * 1000.0
    result = generation["result"]
    prompt = generation["prompt"]
    generated_text = generation["generated_text"]
    normalized = normalize_answer(generated_text)
    gate = answer_gate(normalized, args.expect_contains)
    first_chunk_ms = None
    if first_chunk_at[0] is not None:
        first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0

    created_utc = args.created_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    receipt = {
        "schema_version": "1.0.0",
        "artifact_kind": "lunar_lake_openvino_operator_ask",
        "campaign": "intel-258v-platform",
        "item": "LNL258V-ASK-004",
        "created_utc": created_utc,
        "machine_id": args.machine_id,
        "proof_stage": "operator_candidate_route_executed",
        "requested_backend": expected["selected_backend"],
        "selected_backend": expected["selected_backend"],
        "runtime_api": "openvino_genai",
        "runtime_device": args.device,
        "resolved_device": resolved_device,
        "fallback_used": False,
        "fallback_policy": "strict_no_fallback",
        "backend_lane": expected["backend_lane"],
        "selected_kernel_or_runtime": expected["selected_kernel_or_runtime"],
        "model_family": "qwen",
        "model_architecture": "qwen2",
        "quantization": "INT4_SYM",
        "prompt_template": "qwen2.5",
        "tokenizer_source": "hf_tokenizer_export",
        "route_id": route_id,
        "route": {
            "route_id": route.get("route_id"),
            "route_reason": route.get("route_reason"),
            "acceleration_claim": route.get("acceleration_claim"),
            "answer_gate_evidence": route.get("answer_gate_evidence"),
            "phase_evidence": route.get("phase_evidence"),
        },
        "inputs": {
            "operator_receipt": operator_path.as_posix(),
            "artifact_root": args.artifact_root.as_posix(),
            "model_dir": args.model_dir.as_posix(),
        },
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_model_dir": args.model_dir.as_posix(),
            "model_binary_committed": False,
            "files": {
                name: file_record(args.model_dir / name)
                for name in [
                    "openvino_model.xml",
                    "openvino_model.bin",
                    "openvino_tokenizer.xml",
                    "openvino_tokenizer.bin",
                    "openvino_detokenizer.xml",
                    "openvino_detokenizer.bin",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                    "chat_template.jinja",
                ]
            },
        },
        "prompt_policy": {
            "question": args.question,
            "prompt_template": "qwen2.5",
            "greedy": True,
            "generation_config": {
                "do_sample": False,
                "num_beams": 1,
                "apply_chat_template": True,
                "max_new_tokens": args.max_new_tokens,
            },
            **prompt,
        },
        "output": {
            "generated_text": generated_text,
            "normalized_answer": normalized,
            "generated_token_ids": generation["generated_token_ids"],
            "generated_token_ids_available_from_pipeline": generation[
                "generated_token_ids_available_from_pipeline"
            ],
            "generated_token_ids_source": generation["generated_token_ids_source"],
            "generated_token_count": generation["generated_token_count"],
            "first_streamed_text_chunk_ms": first_chunk_ms,
            "first_streamed_text_chunk": chunks[0]["text"] if chunks else None,
            "streamed_chunks_count": len(chunks),
            "streamed_text": "".join(chunk["text"] for chunk in chunks),
        },
        "answer_gate": gate,
        "timing": {
            "pipeline_construct_wall_ms": pipeline_construct_wall_ms,
            "generation_wall_ms": generation_wall_ms,
            "openvino_perf_metrics": perf_metrics(result),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "openvino": {
                "version": ov.get_version(),
                "available_devices": core.available_devices,
            },
            "openvino_genai": {
                "version": getattr(ov_genai, "__version__", None),
            },
        },
        "verification": {
            "operator_ready": operator.get("operator_ready"),
            "operator_route_validated": True,
            "llmpipeline_constructed": True,
            "generation_ran": True,
            "answer_gate_passed": gate["passed"],
            "fallback_used": False,
            "openvino_perf_metrics_recorded": True,
            "generated_token_ids_available_from_pipeline": True,
            "acceleration_claim": False,
        },
        "claim_boundary": {
            "may_claim": [
                "An explicit Lunar Lake dense SLM OpenVINO candidate route generated a bounded answer with fallback_used=false.",
                "The receipt records selected OpenVINO GenAI backend/runtime/device identity, route reason, answer gate result, and timing fields.",
            ],
            "must_not_claim": [
                "OpenVINO CPU/GPU/NPU speedup or sustained phase performance is proven.",
                "The default Lunar Lake ask route changed away from CPU.",
                "OpenVINO GPU evidence proves native OpenCL execution.",
                "OpenVINO NPU evidence proves native NPU inference outside OpenVINO GenAI.",
                "Dense SLM receipts prove BitNet QK256/I2_S behavior.",
                "Broad dense SLM quality is proven beyond bounded answer gates.",
                "Full BitNet inference works on Arc or NPU.",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
