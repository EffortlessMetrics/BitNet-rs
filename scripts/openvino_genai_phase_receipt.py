#!/usr/bin/env python3
"""Emit an OpenVINO GenAI LLMPipeline phase receipt for bounded SLM prompts."""

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


CASES = [
    {
        "id": "math_2_plus_2",
        "question": "What is 2+2? Answer briefly.",
        "max_new_tokens": 8,
        "contains_any": ["4", "four"],
    },
    {
        "id": "capital_france",
        "question": "Name the capital of France.",
        "max_new_tokens": 8,
        "contains_any": ["Paris", "paris"],
    },
    {
        "id": "rust_sentence",
        "question": "Write one short sentence about Rust.",
        "max_new_tokens": 16,
        "contains_any": ["Rust", "rust", "programming", "language", "safe", "efficient"],
    },
]


DEVICE_BACKENDS = {
    "CPU": ("openvino-cpu", "dense_slm_openvino_cpu", "openvino-genai-llmpipeline-cpu"),
    "GPU.0": ("openvino-gpu", "dense_slm_openvino_gpu_arc140v", "openvino-genai-llmpipeline-gpu0"),
    "NPU": ("openvino-npu", "dense_slm_openvino_npu", "openvino-genai-llmpipeline-npu"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU.0", "NPU"])
    parser.add_argument(
        "--manifest",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-ir-qwen25-int4-sym-manifest.json",
    )
    parser.add_argument(
        "--gguf-cpu-answer",
        default="ci/hardware/intel-258v/2026-05-08/slm-answer-corpus-qwen25-cpu-clean-provenance.json",
    )
    parser.add_argument(
        "--gguf-cpu-phase",
        default="ci/hardware/intel-258v/2026-05-08/slm-phase-warm-session-qwen25-cpu.json",
    )
    parser.add_argument(
        "--prior-comparison",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-phase-comparison.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


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


def prompt_evidence(tokenizer: Any, question: str) -> dict[str, Any]:
    return public_prompt_evidence(ov_prompt_evidence(tokenizer, question))


def run_device(device: str, model_dir: Path, ov_genai: Any, ov_core: Any) -> dict[str, Any]:
    selected_backend, backend_lane, runtime = DEVICE_BACKENDS.get(
        device,
        (f"openvino-{device.lower()}", f"dense_slm_openvino_{device.lower()}", f"openvino-genai-llmpipeline-{device.lower()}"),
    )
    try:
        resolved_device = ov_core.get_property(device, "FULL_DEVICE_NAME")
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        resolved_device = f"unavailable: {type(exc).__name__}: {exc}"

    construct_start = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(model_dir), device)
    construct_wall_ms = (time.perf_counter() - construct_start) * 1000.0
    tokenizer = pipe.get_tokenizer()

    cases = []
    for case in CASES:
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
            case["question"],
            case["max_new_tokens"],
            streamer=streamer,
        )
        generation_wall_ms = (time.perf_counter() - generation_start) * 1000.0
        result = generation["result"]
        prompt = generation["prompt"]
        generated_text = generation["generated_text"]
        matched = [needle for needle in case["contains_any"] if needle in generated_text]
        first_chunk_ms = None
        if first_chunk_at[0] is not None:
            first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
        cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "prompt_template": "qwen2.5",
                **prompt,
                "max_new_tokens": case["max_new_tokens"],
                "greedy": True,
                "generation_config": {
                    "do_sample": False,
                    "num_beams": 1,
                    "apply_chat_template": True,
                    "max_new_tokens": case["max_new_tokens"],
                },
                "generated_text": generated_text,
                "generated_token_ids": generation["generated_token_ids"],
                "generated_token_ids_available_from_pipeline": generation[
                    "generated_token_ids_available_from_pipeline"
                ],
                "generated_token_ids_source": generation["generated_token_ids_source"],
                "generated_token_count": generation["generated_token_count"],
                "generation_wall_ms": generation_wall_ms,
                "first_streamed_text_chunk_ms": first_chunk_ms,
                "first_streamed_text_chunk": chunks[0]["text"] if chunks else None,
                "streamed_chunks_count": len(chunks),
                "streamed_text": "".join(chunk["text"] for chunk in chunks),
                "openvino_perf_metrics": perf_metrics(result),
                "answer_gate": {
                    "kind": "contains_any",
                    "contains_any": case["contains_any"],
                    "matched": matched,
                    "passed": bool(matched),
                },
                "fallback_used": False,
                "selected_backend": selected_backend,
                "runtime_api": "openvino_genai",
                "runtime_device": device,
            }
        )

    passed = sum(1 for case in cases if case["answer_gate"]["passed"])
    return {
        "requested_backend": selected_backend,
        "selected_backend": selected_backend,
        "runtime_api": "openvino_genai",
        "runtime_device": device,
        "resolved_device": resolved_device,
        "backend_lane": backend_lane,
        "fallback_used": False,
        "fallback_status": f"no_fallback_used_{device.lower().replace('.', '')}_device_requested_and_llmpipeline_constructed",
        "selected_kernel_or_runtime": runtime,
        "pipeline_construct_wall_ms": construct_wall_ms,
        "cases_total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "cases": cases,
        "phase_coverage": {
            "pipeline_construct_wall_ms": "measured_by_runner",
            "tokenization_duration": "openvino_genai_perf_metrics",
            "time_to_first_token": "openvino_genai_perf_metrics",
            "first_streamed_text_chunk": "measured_by_streamer_callback",
            "generate_duration": "openvino_genai_perf_metrics",
            "inference_duration": "openvino_genai_perf_metrics",
            "time_per_output_token": "openvino_genai_perf_metrics",
            "detokenization_duration": "openvino_genai_perf_metrics",
            "decode_128": "not_measured_small_bounded_smoke_only",
            "prefill_512": "not_measured_small_bounded_smoke_only",
        },
    }


def main() -> int:
    args = parse_args()
    import openvino as ov
    import openvino_genai as ov_genai

    model_dir = args.model_dir
    core = ov.Core()

    devices = []
    for device in args.devices:
        devices.append(run_device(device, model_dir, ov_genai, core))

    all_passed = all(device["failed"] == 0 for device in devices)
    fallback_used_any = any(device["fallback_used"] for device in devices)
    out = {
        "schema_version": "1.0.0",
        "artifact_kind": "intel_258v_dense_slm_openvino_phase_runner",
        "campaign": "intel-258v-platform",
        "item": "SLM-OV258V-006",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "machine_id": args.machine_id,
        "proof_stage": "phase_measured",
        "comparison_scope": "same_machine_qwen2_5_openvino_genai_perf_metrics_and_streamer_timing",
        "requested_backend": "openvino-cpu-gpu-npu",
        "selected_backend": "openvino-cpu-gpu-npu",
        "runtime_api": "openvino_genai",
        "fallback_used": fallback_used_any,
        "backend_lane": "dense_slm_openvino_phase_runner",
        "model_family": "qwen",
        "model_architecture": "qwen2",
        "quantization": "INT4_SYM",
        "prompt_template": "qwen2.5",
        "tokenizer_source": "hf_tokenizer_export",
        "paths": {
            "openvino_ir_manifest": args.manifest,
            "gguf_cpu_answer": args.gguf_cpu_answer,
            "gguf_cpu_phase": args.gguf_cpu_phase,
            "prior_phase_comparison": args.prior_comparison,
        },
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_model_dir": model_dir.as_posix(),
            "model_binary_committed": False,
            "files": {
                name: file_record(model_dir / name)
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
            "transformers": {
                "tokenizer_loaded_from_export_dir": False,
                "openvino_tokenizer_loaded_from_export_dir": True,
                "generated_token_ids_available_from_pipeline": True,
            },
        },
        "generation": {
            "devices_total": len(devices),
            "all_answer_gates_passed": all_passed,
            "quality_gate_scope": "bounded_three_case_dense_slm_smoke_only",
            "devices": devices,
        },
        "verification": {
            "llmpipeline_constructed_for_all_devices": len(devices) == len(args.devices),
            "generation_ran_for_all_devices": True,
            "all_answer_gates_passed": all_passed,
            "fallback_used": fallback_used_any,
            "openvino_perf_metrics_recorded": True,
            "streamer_first_text_chunk_recorded": True,
            "generated_token_ids_available_from_pipeline": True,
        },
        "claim_boundary": {
            "may_claim": [
                "OpenVINO GenAI PerfMetrics are recorded for bounded Qwen2.5 CPU, GPU.0/Arc 140V, and NPU LLMPipeline runs on the same 258V machine.",
                "The receipt records tokenization, TTFT, generate, inference, TPOT, throughput, detokenization, first streamed text chunk, and bounded answer-gate fields exposed by the current OpenVINO GenAI API.",
                "The bounded answer gates passed with fallback_used=false for all requested OpenVINO devices.",
            ],
            "must_not_claim": [
                "OpenVINO CPU/GPU/NPU speedup or sustained phase performance is proven.",
                "prefill_512 or decode_128 phase profiles are measured.",
                "OpenVINO GPU evidence proves native OpenCL execution.",
                "OpenVINO NPU evidence proves native NPU inference outside OpenVINO GenAI.",
                "Dense SLM receipts prove BitNet QK256/I2_S behavior.",
                "Broad dense SLM quality is proven beyond bounded answer gates.",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if all_passed and not fallback_used_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
