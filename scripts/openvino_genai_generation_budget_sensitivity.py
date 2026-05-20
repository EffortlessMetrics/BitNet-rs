#!/usr/bin/env python3
"""Probe OpenVINO GenAI quality sensitivity to generation-token budgets.

This is a Lunar Lake diagnostic helper. It does not promote routes; it records
whether bounded corpus failures are caused by overgeneration under the fixture
budget or by the first generated tokens already missing the requested answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openvino_genai_corpus_v2 import (
    DEVICE_BACKENDS,
    evaluate_gate,
    evaluate_scoring,
    load_corpus,
    normalize_text,
    perf_metrics,
    prompt_evidence,
    quality_status,
)
from openvino_genai_token_utils import generate_with_direct_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OpenVINO GenAI corpus-v2 generation budget sensitivity.",
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="OpenVINO IR model directory.")
    parser.add_argument("--corpus", type=Path, required=True, help="Lunar Lake answer corpus v2 YAML.")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["CPU", "GPU.0", "NPU"],
        help="OpenVINO runtime devices to probe.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Corpus case id to probe. Defaults to normalized_match cases.",
    )
    parser.add_argument(
        "--max-new-token-variants",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Generation budget variants to compare with each case's fixture budget.",
    )
    parser.add_argument("--json-out", type=Path, required=True, help="Output JSON receipt path.")
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--created-utc", default=None)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def selected_cases(corpus: dict[str, Any], case_ids: list[str]) -> list[dict[str, Any]]:
    cases = list(corpus.get("cases") or [])
    if case_ids:
        selected = [case for case in cases if case.get("id") in set(case_ids)]
    else:
        selected = [
            case
            for case in cases
            if (case.get("scoring") or {}).get("kind") == "normalized_match"
        ]
    return sorted(selected, key=lambda case: str(case.get("id") or ""))


def unique_budgets(case: dict[str, Any], defaults: dict[str, Any], variants: list[int]) -> list[int]:
    fixture_budget = int(case.get("max_new_tokens") or defaults.get("max_new_tokens") or 48)
    budgets = {fixture_budget}
    budgets.update(max(1, int(value)) for value in variants)
    return sorted(budgets)


def run_device(
    device: str,
    model_dir: Path,
    ov_genai: Any,
    ov_core: Any,
    cases: list[dict[str, Any]],
    defaults: dict[str, Any],
    budget_variants: list[int],
) -> dict[str, Any]:
    selected_backend, backend_lane, runtime = DEVICE_BACKENDS.get(
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
    pipe = ov_genai.LLMPipeline(str(model_dir), device)
    construct_wall_ms = (time.perf_counter() - construct_start) * 1000.0
    tokenizer = pipe.get_tokenizer()

    case_results: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        fixture_budget = int(case.get("max_new_tokens") or defaults.get("max_new_tokens") or 48)
        prompt = prompt_evidence(tokenizer, question)
        variants_out: list[dict[str, Any]] = []
        for max_new_tokens in unique_budgets(case, defaults, budget_variants):
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
                question,
                max_new_tokens,
                streamer=streamer,
            )
            generation_wall_ms = (time.perf_counter() - generation_start) * 1000.0
            result = generation["result"]
            generated_text = generation["generated_text"]
            gate = evaluate_gate(case.get("gate"), generated_text)
            scoring = evaluate_scoring(case.get("scoring"), generated_text)
            quality = quality_status(gate, scoring, generated_text)
            first_chunk_ms = None
            if first_chunk_at[0] is not None:
                first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
            variants_out.append(
                {
                    "max_new_tokens": max_new_tokens,
                    "is_fixture_budget": max_new_tokens == fixture_budget,
                    "generated_text": generated_text,
                    "decoded_preview": generated_text[:240],
                    "normalized_output": normalize_text(generated_text),
                    "generated_token_ids": generation["generated_token_ids"],
                    "generated_token_ids_available_from_pipeline": generation[
                        "generated_token_ids_available_from_pipeline"
                    ],
                    "generated_token_ids_source": generation["generated_token_ids_source"],
                    "generated_token_count": generation["generated_token_count"],
                    "answer_gate": gate,
                    "scoring": scoring,
                    "quality": quality,
                    "timing": {
                        "generation_wall_ms": generation_wall_ms,
                        "first_streamed_text_chunk_ms": first_chunk_ms,
                        "first_streamed_text_chunk": chunks[0]["text"] if chunks else None,
                        "streamed_chunks_count": len(chunks),
                        "openvino_perf_metrics": perf_metrics(result),
                    },
                    "status": "passed" if quality["passed"] else "quality_failed",
                }
            )

        fixture = next((variant for variant in variants_out if variant["is_fixture_budget"]), None)
        passing = [variant for variant in variants_out if variant["quality"]["passed"]]
        first_passing = passing[0] if passing else None
        if fixture and fixture["quality"]["passed"]:
            blocker_class = "fixture_budget_passes"
        elif first_passing:
            blocker_class = "fixture_budget_overgenerates_but_smaller_budget_passes"
        else:
            blocker_class = "no_budget_variant_passes"
        case_results.append(
            {
                "id": case["id"],
                "profile": case.get("profile", "unknown"),
                "category": case.get("category", "unknown"),
                "question": question,
                "prompt_template": defaults.get("prompt_template", "qwen2.5"),
                "prompt": {
                    "template_family": defaults.get("prompt_template", "qwen2.5"),
                    **prompt,
                },
                "fixture_max_new_tokens": fixture_budget,
                "budget_variants_tested": [variant["max_new_tokens"] for variant in variants_out],
                "fixture_budget_passed": bool(fixture and fixture["quality"]["passed"]),
                "any_budget_passed": bool(first_passing),
                "first_passing_budget": first_passing["max_new_tokens"] if first_passing else None,
                "blocker_class": blocker_class,
                "variants": variants_out,
            }
        )

    summary = summarize_device(case_results)
    return {
        "requested_backend": selected_backend,
        "selected_backend": selected_backend,
        "runtime_api": "openvino_genai",
        "runtime_device": device,
        "resolved_device": resolved_device,
        "backend_lane": backend_lane,
        "fallback_used": False,
        "selected_kernel_or_runtime": runtime,
        "pipeline_construct_wall_ms": construct_wall_ms,
        "promotion_status": "candidate_only_not_promoted",
        "route_promotion_changed": False,
        "summary": summary,
        "cases": case_results,
    }


def summarize_device(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_blocker: dict[str, int] = defaultdict(int)
    for case in case_results:
        by_blocker[case["blocker_class"]] += 1
    return {
        "cases_total": len(case_results),
        "fixture_budget_passed": sum(1 for case in case_results if case["fixture_budget_passed"]),
        "any_budget_passed": sum(1 for case in case_results if case["any_budget_passed"]),
        "blocker_classes": dict(sorted(by_blocker.items())),
    }


def main() -> int:
    args = parse_args()
    created = args.created_utc or utc_now()
    corpus = load_corpus(args.corpus)
    defaults = corpus.get("defaults") or {}
    cases = selected_cases(corpus, args.case_id)
    if not cases:
        raise SystemExit("no corpus cases selected for generation budget sensitivity probe")

    try:
        import openvino as ov
        import openvino_genai as ov_genai
    except Exception as exc:  # pragma: no cover - depends on local OpenVINO install.
        raise SystemExit(f"OpenVINO GenAI import failed: {type(exc).__name__}: {exc}") from exc

    ov_core = ov.Core()
    devices = [
        run_device(
            device=device,
            model_dir=args.model_dir,
            ov_genai=ov_genai,
            ov_core=ov_core,
            cases=cases,
            defaults=defaults,
            budget_variants=args.max_new_token_variants,
        )
        for device in args.devices
    ]

    aggregate_blockers: dict[str, int] = defaultdict(int)
    for device in devices:
        for key, value in (device.get("summary") or {}).get("blocker_classes", {}).items():
            aggregate_blockers[key] += int(value)

    receipt = {
        "schema_version": "1.0.0",
        "artifact_kind": "intel_258v_dense_slm_openvino_generation_budget_sensitivity",
        "campaign": "intel-258v-platform",
        "item": "LNL258V-OV-QUAL-005",
        "created_utc": created,
        "machine_id": args.machine_id,
        "proof_stage": "candidate_route_generation_budget_diagnostic_no_promotion_change",
        "comparison_scope": "same_machine_qwen2_5_openvino_candidate_route_exact_answer_budget_sensitivity",
        "requested_backend": "openvino-cpu-gpu-npu",
        "selected_backend": "openvino-cpu-gpu-npu",
        "runtime_api": "openvino_genai",
        "fallback_used": False,
        "backend_lane": "dense_slm_openvino_candidate_routes",
        "model_family": "qwen",
        "model_architecture": "qwen2",
        "quantization": "INT4_SYM",
        "prompt_template": "qwen2.5",
        "tokenizer_source": "hf_tokenizer_export",
        "promotion_status": "candidate_only_not_promoted",
        "route_promotion_changed": False,
        "diagnostic_policy": {
            "selected_cases": "explicit case ids or normalized_match corpus cases",
            "fixture_budget": "case max_new_tokens from answer-corpus-v2",
            "budget_variants": sorted(set(max(1, int(value)) for value in args.max_new_token_variants)),
            "pass_interpretation": "smaller-budget pass after fixture failure indicates overgeneration sensitivity, not route promotion",
        },
        "paths": {
            "answer_corpus_v2": str(args.corpus).replace("\\", "/"),
            "openvino_ir_model_dir": str(args.model_dir).replace("\\", "/"),
        },
        "cases_selected": [case["id"] for case in cases],
        "devices": devices,
        "summary": {
            "devices_total": len(devices),
            "cases_per_device": len(cases),
            "aggregate_blocker_classes": dict(sorted(aggregate_blockers.items())),
            "fallback_used": False,
            "route_promotion_changed": False,
        },
        "claim_boundary": {
            "new_inference_executed": True,
            "route_promotion": False,
            "speedup_claim": False,
            "power_advantage_claim": False,
            "arc_acceleration_claim": False,
            "npu_acceleration_claim": False,
            "full_bitnet_accelerator_inference_claim": False,
            "bitnet_qk256_i2s_behavior_change": False,
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
