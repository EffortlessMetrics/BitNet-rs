#!/usr/bin/env python3
"""Probe OpenVINO GenAI NPU cache behavior for Lunar Lake dense Qwen.

The parent process launches two child Python processes against one explicit
OpenVINO CACHE_DIR. Each child constructs an NPU LLMPipeline and runs one
bounded ask, so the receipt can compare first-process and second-process cold
startup without confusing it with same-process resident behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openvino_genai_token_utils import generate_with_direct_token_ids
from openvino_genai_token_utils import prompt_evidence as ov_prompt_evidence
from openvino_genai_token_utils import public_prompt_evidence


QUESTION = "What is 2+2? Answer briefly."
EXPECTED = "4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("target/openvino-cache/lnl258v-npu-cache-probe-runs"),
        help="Directory for child process sidecar receipts. Not intended for commit.",
    )
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--created-utc")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--expect-contains", default=EXPECTED)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--material-improvement-ratio", type=float, default=0.75)
    parser.add_argument("--material-improvement-ms", type=float, default=2000.0)
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-iteration", help=argparse.SUPPRESS)
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


def cache_snapshot(cache_dir: Path) -> dict[str, Any]:
    files = []
    if cache_dir.exists():
        for path in sorted(p for p in cache_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(cache_dir).as_posix()
            stat = path.stat()
            files.append(
                {
                    "path": rel,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path) if stat.st_size <= 1024 * 1024 else None,
                }
            )
    return {
        "cache_dir": cache_dir.as_posix(),
        "exists": cache_dir.exists(),
        "file_count": len(files),
        "total_bytes": sum(int(file["bytes"]) for file in files),
        "files": files,
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


def prompt_evidence(tokenizer: Any, question: str) -> dict[str, Any]:
    return public_prompt_evidence(ov_prompt_evidence(tokenizer, question))


def normalize_answer(text: str) -> str:
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def run_child(args: argparse.Namespace) -> int:
    if args.device != "NPU":
        raise SystemExit("this cache probe is scoped to --device NPU")
    import openvino as ov
    import openvino_genai as ov_genai

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    try:
        resolved_device = core.get_property(args.device, "FULL_DEVICE_NAME")
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        resolved_device = f"unavailable: {type(exc).__name__}: {exc}"

    construct_start = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(args.model_dir), args.device, {"CACHE_DIR": str(args.cache_dir)})
    construct_wall_ms = (time.perf_counter() - construct_start) * 1000.0
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
    first_chunk_ms = None
    if first_chunk_at[0] is not None:
        first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
    receipt = {
        "iteration": args.child_iteration,
        "runtime_api": "openvino_genai",
        "runtime_device": args.device,
        "resolved_device": resolved_device,
        "selected_backend": "openvino-npu",
        "fallback_used": False,
        "cache_dir": args.cache_dir.as_posix(),
        "pipeline_construct_wall_ms": construct_wall_ms,
        "prompt": prompt,
        "question": args.question,
        "max_new_tokens": args.max_new_tokens,
        "generated_text": generated_text,
        "generated_token_ids": generation["generated_token_ids"],
        "generated_token_ids_available_from_pipeline": generation[
            "generated_token_ids_available_from_pipeline"
        ],
        "generated_token_ids_source": generation["generated_token_ids_source"],
        "generated_token_count": generation["generated_token_count"],
        "generation_wall_ms": generation_wall_ms,
        "first_streamed_text_chunk_ms": first_chunk_ms,
        "streamed_chunks_count": len(chunks),
        "streamed_text": "".join(chunk["text"] for chunk in chunks),
        "openvino_perf_metrics": perf_metrics(result),
        "answer_gate": {
            "kind": "contains",
            "expected": args.expect_contains,
            "passed": args.expect_contains in normalized,
            "normalized": normalized,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


def run_child_process(args: argparse.Namespace, iteration: str, child_out: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--child-iteration",
        iteration,
        "--model-dir",
        str(args.model_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--json-out",
        str(child_out),
        "--device",
        args.device,
        "--question",
        args.question,
        "--expect-contains",
        args.expect_contains,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    wall_ms = (time.perf_counter() - started) * 1000.0
    child = json.loads(child_out.read_text(encoding="utf-8")) if child_out.exists() else None
    return {
        "iteration": iteration,
        "command": command,
        "process_returncode": completed.returncode,
        "process_wall_ms": wall_ms,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "child_receipt": child,
    }


def main() -> int:
    args = parse_args()
    if args.child_run:
        return run_child(args)
    if args.device != "NPU":
        raise SystemExit("this cache probe is scoped to --device NPU")

    import openvino as ov
    import openvino_genai as ov_genai

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    initial_snapshot = cache_snapshot(args.cache_dir)
    child_dir = args.work_dir
    child_dir.mkdir(parents=True, exist_ok=True)
    first_child = child_dir / "first-process.json"
    second_child = child_dir / "second-process.json"

    first = run_child_process(args, "first_process", first_child)
    after_first_snapshot = cache_snapshot(args.cache_dir)
    second = run_child_process(args, "second_process", second_child)
    after_second_snapshot = cache_snapshot(args.cache_dir)

    first_construct = None
    second_construct = None
    if first.get("child_receipt"):
        first_construct = first["child_receipt"].get("pipeline_construct_wall_ms")
    if second.get("child_receipt"):
        second_construct = second["child_receipt"].get("pipeline_construct_wall_ms")
    improvement_ms = None
    improvement_ratio = None
    cache_effective = False
    if first_construct is not None and second_construct is not None and first_construct > 0:
        improvement_ms = float(first_construct) - float(second_construct)
        improvement_ratio = float(second_construct) / float(first_construct)
        cache_effective = (
            improvement_ms >= args.material_improvement_ms
            or improvement_ratio <= args.material_improvement_ratio
        )

    first_passed = first.get("child_receipt", {}).get("answer_gate", {}).get("passed") is True
    second_passed = second.get("child_receipt", {}).get("answer_gate", {}).get("passed") is True
    fallback_used = any(
        run.get("child_receipt", {}).get("fallback_used") is True for run in [first, second]
    )
    cache_files_created = (
        after_first_snapshot["file_count"] > initial_snapshot["file_count"]
        or after_first_snapshot["total_bytes"] > initial_snapshot["total_bytes"]
    )
    cache_files_reused_or_stable = (
        after_second_snapshot["file_count"] >= after_first_snapshot["file_count"]
        and after_second_snapshot["total_bytes"] >= after_first_snapshot["total_bytes"]
    )
    cache_hit_evidence = "not_available"
    if cache_files_created and cache_effective:
        cache_hit_evidence = "cache_files_created_then_second_process_timing_improved"
    elif cache_files_created:
        cache_hit_evidence = "cache_files_created_but_second_process_timing_not_materially_improved"
    elif initial_snapshot["file_count"] > 0 and cache_effective:
        cache_hit_evidence = "preexisting_cache_files_and_second_process_timing_improved"

    created_utc = args.created_utc or utc_now()
    receipt = {
        "schema_version": "1.0.0",
        "artifact_kind": "lunar_lake_openvino_npu_cache_experiment",
        "campaign": "intel-258v-platform",
        "item": "LNL258V-NPU-CACHE-001",
        "created_utc": created_utc,
        "machine_id": args.machine_id,
        "proof_stage": "candidate_route_cache_hit_miss_experiment_no_promotion_change",
        "artifact_root": "ci/hardware/intel-258v/2026-05-08",
        "comparison_scope": "two_separate_openvino_genai_npu_processes_with_one_cache_dir",
        "route_id": "dense_slm_openvino_npu_candidate",
        "requested_backend": "openvino-npu",
        "selected_backend": "openvino-npu",
        "runtime_api": "openvino_genai",
        "runtime_device": args.device,
        "backend_lane": "dense_slm_openvino_npu",
        "selected_kernel_or_runtime": "openvino-genai-llmpipeline-npu",
        "fallback_used": fallback_used,
        "model": {
            "source": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_path": args.model_dir.as_posix(),
            "architecture": "qwen2",
            "openvino_ir": {
                "xml": file_record(args.model_dir / "openvino_model.xml"),
                "bin": file_record(args.model_dir / "openvino_model.bin"),
            },
        },
        "cache": {
            "cache_dir": args.cache_dir.as_posix(),
            "cache_enabled": True,
            "cache_writable": args.cache_dir.exists() and args.cache_dir.is_dir(),
            "cache_hit_evidence": cache_hit_evidence,
            "cache_hit_runtime_metric_available": False,
            "cache_files_created": cache_files_created,
            "cache_files_reused_or_stable": cache_files_reused_or_stable,
            "material_improvement_ratio_threshold": args.material_improvement_ratio,
            "material_improvement_ms_threshold": args.material_improvement_ms,
            "cache_effective_by_timing": cache_effective,
            "initial_snapshot": initial_snapshot,
            "after_first_process_snapshot": after_first_snapshot,
            "after_second_process_snapshot": after_second_snapshot,
        },
        "process_runs": [first, second],
        "comparison": {
            "first_pipeline_construct_wall_ms": first_construct,
            "second_pipeline_construct_wall_ms": second_construct,
            "second_to_first_construct_ratio": improvement_ratio,
            "construct_improvement_ms": improvement_ms,
            "first_answer_gate_passed": first_passed,
            "second_answer_gate_passed": second_passed,
            "cache_experiment_ready": (
                first.get("process_returncode") == 0
                and second.get("process_returncode") == 0
                and first_passed
                and second_passed
                and fallback_used is False
            ),
            "classification": (
                "cache_materially_reduces_pipeline_construct"
                if cache_effective
                else "cache_not_materially_proven_for_pipeline_construct"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "openvino_version": getattr(ov, "__version__", None),
            "openvino_genai_version": getattr(ov_genai, "__version__", None),
        },
        "generated_token_visibility": {
            "direct_generated_token_ids_available": True,
            "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
        },
        "claim_boundary": {
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
                "OpenVINO runtime cache hit is directly exposed by a metric",
                "Full BitNet inference works on Arc or NPU",
                "Packed QK256 decode works on Arc or NPU",
                "Dense SLM receipts prove BitNet QK256/I2_S behavior",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": args.json_out.as_posix(),
                "cache_effective_by_timing": cache_effective,
                "first_construct_ms": first_construct,
                "second_construct_ms": second_construct,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
