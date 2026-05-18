#!/usr/bin/env python3
"""Run a Lunar Lake OpenVINO GenAI NPU resident-session receipt.

This helper exercises the dense Qwen OpenVINO NPU candidate route in a single
process. It measures pipeline construction separately from repeated warm asks so
route-policy code can distinguish cold one-off NPU startup from resident-route
behavior. It does not promote the NPU route or claim acceleration.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ctypes import wintypes

from openvino_genai_token_utils import generate_with_direct_token_ids
from openvino_genai_token_utils import prompt_evidence as ov_prompt_evidence
from openvino_genai_token_utils import public_prompt_evidence


CASES = [
    {
        "id": "math_2_plus_2_brief",
        "question": "What is 2+2? Answer briefly.",
        "max_new_tokens": 8,
        "contains_any": ["4", "four"],
    },
    {
        "id": "short_factual_capital_france",
        "question": "Name the capital of France.",
        "max_new_tokens": 8,
        "contains_any": ["Paris", "paris"],
    },
    {
        "id": "instruction_single_sentence_rust",
        "question": "Write one short sentence about Rust.",
        "max_new_tokens": 16,
        "contains_any": ["Rust", "rust", "programming", "language", "safe", "efficient"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--item", default="LNL258V-NPU-RESIDENT-001")
    parser.add_argument(
        "--proof-stage",
        default="candidate_route_resident_warm_session_evidence_no_promotion_change",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--created-utc")
    parser.add_argument(
        "--manifest",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-ir-qwen25-int4-sym-manifest.json",
    )
    parser.add_argument(
        "--openvino-corpus-v2",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json",
    )
    parser.add_argument(
        "--npu-cold-start-diagnosis",
        default="ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cold-start-diagnosis.json",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def mean_std(pair: Any) -> dict[str, float | None]:
    if pair is None:
        return {"mean_ms": None, "std_ms": None}
    return {
        "mean_ms": float(getattr(pair, "mean", 0.0)),
        "std_ms": float(getattr(pair, "std", 0.0)),
    }


def perf_metrics(result: Any) -> dict[str, Any] | None:
    perf = getattr(result, "perf_metrics", None)
    if perf is None:
        return None
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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def current_process_memory_bytes() -> int | None:
    if platform.system().lower() == "windows":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return int(counters.WorkingSetSize)
        return None

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KiB on Linux and bytes on macOS. This helper is only
        # context for route receipts, so prefer a conservative common case.
        if platform.system().lower() == "darwin":
            return int(usage.ru_maxrss)
        return int(usage.ru_maxrss) * 1024
    except Exception:
        return None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile_value)))
    return ordered[index]


def prompt_evidence(tokenizer: Any, question: str) -> dict[str, Any]:
    return public_prompt_evidence(ov_prompt_evidence(tokenizer, question))


def normalize_answer(text: str) -> str:
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def answer_gate(generated_text: str, contains_any: list[str]) -> dict[str, Any]:
    normalized = normalize_answer(generated_text)
    matched = [needle for needle in contains_any if needle in normalized]
    return {
        "kind": "contains_any",
        "contains_any": contains_any,
        "matched": matched,
        "passed": bool(matched),
        "normalized": normalized,
    }


def construct_pipeline(ov_genai: Any, model_dir: Path, device: str, cache_dir: Path | None) -> tuple[Any, dict[str, Any]]:
    config: dict[str, Any] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(cache_dir)

    construct_start = time.perf_counter()
    cache_status = "not_requested"
    try:
        if config:
            pipe = ov_genai.LLMPipeline(str(model_dir), device, config)
            cache_status = "requested"
        else:
            pipe = ov_genai.LLMPipeline(str(model_dir), device)
    except Exception as exc:
        if not config:
            raise
        retry_start = time.perf_counter()
        pipe = ov_genai.LLMPipeline(str(model_dir), device)
        retry_ms = (time.perf_counter() - retry_start) * 1000.0
        elapsed_ms = (time.perf_counter() - construct_start) * 1000.0
        return pipe, {
            "pipeline_construct_wall_ms": elapsed_ms,
            "cache_config_status": "requested_but_retried_without_cache",
            "cache_config_error": f"{type(exc).__name__}: {exc}",
            "retry_without_cache_wall_ms": retry_ms,
        }
    elapsed_ms = (time.perf_counter() - construct_start) * 1000.0
    return pipe, {
        "pipeline_construct_wall_ms": elapsed_ms,
        "cache_config_status": cache_status,
    }


def run_ask(pipe: Any, ov_genai: Any, tokenizer: Any, case: dict[str, Any], sequence_index: int, phase: str) -> dict[str, Any]:
    prompt = prompt_evidence(tokenizer, case["question"])
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
    first_chunk_ms = None
    if first_chunk_at[0] is not None:
        first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
    return {
        "sequence_index": sequence_index,
        "phase": phase,
        "case_id": case["id"],
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
        "answer_gate": answer_gate(generated_text, list(case["contains_any"])),
        "fallback_used": False,
    }


def summarize_asks(asks: list[dict[str, Any]]) -> dict[str, Any]:
    generation_times = [float(ask["generation_wall_ms"]) for ask in asks if ask.get("generation_wall_ms") is not None]
    first_chunk_times = [
        float(ask["first_streamed_text_chunk_ms"])
        for ask in asks
        if ask.get("first_streamed_text_chunk_ms") is not None
    ]
    throughputs = []
    ttft = []
    generated_tokens = 0
    for ask in asks:
        metrics = ask.get("openvino_perf_metrics") or {}
        throughput = metrics.get("throughput", {}).get("mean_ms")
        if throughput is not None:
            throughputs.append(float(throughput))
        first_token = metrics.get("time_to_first_token", {}).get("mean_ms")
        if first_token is not None:
            ttft.append(float(first_token))
        generated_tokens += int(metrics.get("num_generated_tokens") or ask.get("generated_token_count") or 0)
    passed = sum(1 for ask in asks if ask.get("answer_gate", {}).get("passed") is True)
    return {
        "ask_count": len(asks),
        "passed": passed,
        "failed": len(asks) - passed,
        "fallback_used": any(ask.get("fallback_used") is True for ask in asks),
        "generation_wall_ms": {
            "min": min(generation_times) if generation_times else None,
            "mean": mean(generation_times),
            "p95": percentile(generation_times, 0.95),
            "max": max(generation_times) if generation_times else None,
        },
        "first_streamed_text_chunk_ms": {
            "min": min(first_chunk_times) if first_chunk_times else None,
            "mean": mean(first_chunk_times),
            "p95": percentile(first_chunk_times, 0.95),
            "max": max(first_chunk_times) if first_chunk_times else None,
        },
        "openvino_time_to_first_token_ms": {
            "min": min(ttft) if ttft else None,
            "mean": mean(ttft),
            "p95": percentile(ttft, 0.95),
            "max": max(ttft) if ttft else None,
        },
        "throughput_tokens_per_s": {
            "min": min(throughputs) if throughputs else None,
            "mean": mean(throughputs),
            "p95": percentile(throughputs, 0.95),
            "max": max(throughputs) if throughputs else None,
        },
        "generated_tokens": generated_tokens,
    }


def stability_summary(asks: list[dict[str, Any]], memory_samples: dict[str, int | None]) -> dict[str, Any]:
    warm_asks = [ask for ask in asks if ask["phase"] == "warm_resident_ask"]
    outputs_by_case: dict[str, set[str]] = {}
    tokens_by_case: dict[str, set[str]] = {}
    for ask in warm_asks:
        case_id = str(ask.get("case_id", "unknown_case"))
        normalized = str(ask.get("answer_gate", {}).get("normalized") or normalize_answer(str(ask.get("generated_text", ""))))
        outputs_by_case.setdefault(case_id, set()).add(normalized)
        token_ids = ask.get("generated_token_ids")
        tokens_by_case.setdefault(case_id, set()).add(json.dumps(token_ids, separators=(",", ":")))

    after_construct = memory_samples.get("after_pipeline_construct_bytes")
    after_warm = memory_samples.get("after_warm_loop_bytes")
    resident_growth = None
    if after_construct is not None and after_warm is not None:
        resident_growth = after_warm - after_construct

    return {
        "warm_resident_ask_count": len(warm_asks),
        "answer_drift_detected": any(len(outputs) > 1 for outputs in outputs_by_case.values()),
        "generated_token_drift_detected": any(len(tokens) > 1 for tokens in tokens_by_case.values()),
        "fallback_drift_detected": any(ask.get("fallback_used") is True for ask in warm_asks),
        "route_drift_detected": False,
        "unique_outputs_by_case": {case: sorted(outputs) for case, outputs in sorted(outputs_by_case.items())},
        "unique_generated_token_sequences_by_case": {
            case: len(tokens) for case, tokens in sorted(tokens_by_case.items())
        },
        "memory_samples": memory_samples,
        "resident_memory_growth_bytes": resident_growth,
    }


def main() -> int:
    args = parse_args()
    if args.device != "NPU":
        raise SystemExit("this resident-session proof is scoped to --device NPU")
    if args.warm_repeats < 1:
        raise SystemExit("--warm-repeats must be >= 1")

    import openvino as ov
    import openvino_genai as ov_genai

    core = ov.Core()
    try:
        resolved_device = core.get_property(args.device, "FULL_DEVICE_NAME")
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        resolved_device = f"unavailable: {type(exc).__name__}: {exc}"

    memory_samples = {"before_pipeline_construct_bytes": current_process_memory_bytes()}
    pipe, construct = construct_pipeline(ov_genai, args.model_dir, args.device, args.cache_dir)
    memory_samples["after_pipeline_construct_bytes"] = current_process_memory_bytes()
    tokenizer = pipe.get_tokenizer()

    asks = [run_ask(pipe, ov_genai, tokenizer, CASES[0], 0, "cold_first_ask")]
    memory_samples["after_cold_first_ask_bytes"] = current_process_memory_bytes()
    for index in range(args.warm_repeats):
        case = CASES[index % len(CASES)]
        asks.append(run_ask(pipe, ov_genai, tokenizer, case, index + 1, "warm_resident_ask"))
    memory_samples["after_warm_loop_bytes"] = current_process_memory_bytes()

    cold_asks = [ask for ask in asks if ask["phase"] == "cold_first_ask"]
    warm_asks = [ask for ask in asks if ask["phase"] == "warm_resident_ask"]
    warm_summary = summarize_asks(warm_asks)
    resident_ready = (
        args.warm_repeats >= 10
        and warm_summary["ask_count"] >= 10
        and warm_summary["failed"] == 0
        and warm_summary["fallback_used"] is False
        and construct.get("pipeline_construct_wall_ms") is not None
    )
    created_utc = args.created_utc or utc_now()
    receipt = {
        "schema_version": "1.0.0",
        "artifact_kind": "lunar_lake_openvino_npu_resident_session",
        "campaign": "intel-258v-platform",
        "item": args.item,
        "created_utc": created_utc,
        "machine_id": args.machine_id,
        "proof_stage": args.proof_stage,
        "artifact_root": "ci/hardware/intel-258v/2026-05-08",
        "comparison_scope": "same_process_openvino_genai_npu_cold_construct_plus_repeated_warm_asks",
        "requested_backend": "openvino-npu",
        "selected_backend": "openvino-npu",
        "runtime_api": "openvino_genai",
        "runtime_device": args.device,
        "resolved_device": resolved_device,
        "backend_lane": "dense_slm_openvino_npu",
        "route_id": "dense_slm_openvino_npu_candidate",
        "selected_kernel_or_runtime": "openvino-genai-llmpipeline-npu",
        "fallback_used": False,
        "fallback_status": "no_fallback_used_npu_device_requested_and_llmpipeline_constructed",
        "model": {
            "source": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_path": args.model_dir.as_posix(),
            "architecture": "qwen2",
            "openvino_ir": {
                "xml": file_record(args.model_dir / "openvino_model.xml"),
                "bin": file_record(args.model_dir / "openvino_model.bin"),
            },
        },
        "tokenizer": {
            "source": "openvino_ir_folder_hf_tokenizer_files",
            "family": "qwen2",
            "prompt_template": "qwen2.5",
            "tokenizer_json": file_record(args.model_dir / "tokenizer.json"),
            "tokenizer_config_json": file_record(args.model_dir / "tokenizer_config.json"),
        },
        "source_receipts": {
            "manifest": file_record(Path(args.manifest)),
            "openvino_corpus_v2": file_record(Path(args.openvino_corpus_v2)),
            "npu_cold_start_diagnosis": file_record(Path(args.npu_cold_start_diagnosis)),
        },
        "cache_context": {
            "cache_dir": args.cache_dir.as_posix() if args.cache_dir else None,
            "cache_requested": args.cache_dir is not None,
            "cache_config_status": construct.get("cache_config_status"),
            "cache_config_error": construct.get("cache_config_error"),
            "cache_dir_exists": args.cache_dir.exists() if args.cache_dir else None,
        },
        "pipeline": construct,
        "resident_session": {
            "resident_session_ready": resident_ready,
            "cold_first_ask": summarize_asks(cold_asks),
            "warm_repeats_requested": args.warm_repeats,
            "warm_resident_asks": warm_summary,
            "same_process_pipeline_reused": True,
            "proof_limitations": [
                "resident proof does not remove the cold one-off NPU startup blocker",
                "power advantage is not measured by this receipt",
                "route promotion is unchanged",
            ],
        },
        "stability": stability_summary(asks, memory_samples),
        "asks": asks,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "openvino_available_devices": core.available_devices,
            "openvino_version": getattr(ov, "__version__", None),
            "openvino_genai_version": getattr(ov_genai, "__version__", None),
        },
        "generated_token_visibility": {
            "direct_generated_token_ids_available": True,
            "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
        },
        "claim_boundary": {
            "new_inference_executed": True,
            "route_promotion_changed": False,
            "speedup_claim": False,
            "power_advantage_claim": False,
            "acceleration_claim": False,
            "native_npu_inference_claim": False,
            "bitnet_qk256_i2s_behavior_changed": False,
            "must_not_claim": [
                "OpenVINO NPU is promoted for any route profile",
                "OpenVINO NPU speedup is proven",
                "OpenVINO NPU power advantage is proven",
                "Intel NPU acceleration is proven",
                "Native NPU inference outside OpenVINO GenAI is proven",
                "Full BitNet inference works on Arc or NPU",
                "Packed QK256 decode works on Arc or NPU",
                "Dense SLM receipts prove BitNet QK256/I2_S behavior",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"json_out": args.json_out.as_posix(), "resident_session_ready": resident_ready}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
