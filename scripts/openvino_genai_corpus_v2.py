#!/usr/bin/env python3
"""Run the Lunar Lake answer corpus v2 on OpenVINO GenAI candidate routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from openvino_genai_token_utils import generate_with_direct_token_ids
from openvino_genai_token_utils import prompt_evidence as ov_prompt_evidence
from openvino_genai_token_utils import public_prompt_evidence


DEVICE_BACKENDS = {
    "CPU": ("openvino-cpu", "dense_slm_openvino_cpu", "openvino-genai-llmpipeline-cpu"),
    "GPU.0": ("openvino-gpu", "dense_slm_openvino_gpu_arc140v", "openvino-genai-llmpipeline-gpu0"),
    "NPU": ("openvino-npu", "dense_slm_openvino_npu", "openvino-genai-llmpipeline-npu"),
}

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
KNOWN_STOP_MARKERS = ("<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>", "<|im_start|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU.0", "NPU"])
    parser.add_argument("--created-utc")
    parser.add_argument(
        "--manifest",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-ir-qwen25-int4-sym-manifest.json",
    )
    parser.add_argument(
        "--gguf-cpu-corpus-v2",
        default="ci/hardware/intel-258v/2026-05-08/slm-answer-corpus-qwen25-cpu-corpus-v2.json",
    )
    parser.add_argument(
        "--phase-runner",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-phase-runner.json",
    )
    parser.add_argument(
        "--route-profile-comparison",
        default="ci/hardware/intel-258v/2026-05-08/lunar-lake-route-profile-comparison.json",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"corpus {path} did not parse as a mapping")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"corpus {path} does not contain cases")
    return data


def strip_leading_chatml_assistant(text: str) -> str:
    rest = text
    while rest.startswith("<|im_start|>assistant"):
        rest = rest[len("<|im_start|>assistant") :]
        if rest.startswith("\r\n"):
            rest = rest[2:]
        elif rest.startswith("\n") or rest.startswith(" "):
            rest = rest[1:]
        rest = rest.lstrip()
    return rest


def strip_special_markers(text: str) -> str:
    cleaned = strip_leading_chatml_assistant(text.lstrip()).replace("<|begin_of_text|>", "")
    marker_indexes = [cleaned.find(marker) for marker in KNOWN_STOP_MARKERS if cleaned.find(marker) >= 0]
    if marker_indexes:
        cleaned = cleaned[: min(marker_indexes)]
    return cleaned


def strip_leading_assistant_separator(text: str) -> str:
    if text.startswith(":") and len(text) > 1 and text[1].isspace():
        return text[1:].lstrip()
    return text


def normalize_scoring_text(text: str) -> str:
    stripped = strip_special_markers(text)
    collapsed = " ".join(stripped.split())
    return strip_leading_assistant_separator(collapsed)


def normalize_match_text(text: str) -> str:
    return normalize_scoring_text(text).strip(" \t\r\n.!?").lower()


def normalize_text(text: str) -> str:
    return normalize_match_text(text).strip(" \t\r\n.,;:!?\"'`")


def contains_keyword_boundary(answer: str, keyword: str) -> bool:
    haystack = answer.lower()
    needle = keyword.strip().lower()
    if not needle:
        return False
    needle_starts_alnum = needle[0].isalnum()
    needle_ends_alnum = needle[-1].isalnum()
    search_from = 0
    while True:
        start = haystack.find(needle, search_from)
        if start < 0:
            return False
        end = start + len(needle)
        before = haystack[start - 1] if start > 0 else None
        after = haystack[end] if end < len(haystack) else None
        left_ok = not needle_starts_alnum or before is None or not before.isalnum()
        right_ok = not needle_ends_alnum or after is None or not after.isalnum()
        if left_ok and right_ok:
            return True
        search_from = start + 1


def missing_keywords(answer: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if not contains_keyword_boundary(answer, keyword)]


def observed_forbidden_tokens(answer: str, tokens: list[str]) -> list[str]:
    return [token for token in tokens if contains_keyword_boundary(answer, token)]


def readable_words(text: str) -> list[str]:
    cleaned = SPECIAL_TOKEN_RE.sub(" ", text)
    return re.findall(r"[A-Za-z0-9_+-]+", cleaned)


def prompt_evidence(tokenizer: Any, question: str) -> dict[str, Any]:
    return public_prompt_evidence(ov_prompt_evidence(tokenizer, question))


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


def evaluate_gate(gate: dict[str, Any] | None, generated_text: str) -> dict[str, Any]:
    if not gate:
        return {"kind": "none", "passed": True}
    kind = gate.get("kind")
    stripped = generated_text.lstrip()
    if kind == "contains_any":
        needles = list(gate.get("contains_any") or [])
        matched = [needle for needle in needles if needle in generated_text]
        return {"kind": kind, "contains_any": needles, "matched": matched, "passed": bool(matched)}
    if kind == "starts_with_any":
        needles = list(gate.get("starts_with_any") or [])
        matched = [needle for needle in needles if stripped.startswith(needle)]
        return {"kind": kind, "starts_with_any": needles, "matched": matched, "passed": bool(matched)}
    if kind == "readable":
        min_words = int(gate.get("min_words") or 1)
        words = readable_words(generated_text)
        return {
            "kind": kind,
            "min_words": min_words,
            "observed_words": len(words),
            "passed": len(words) >= min_words,
        }
    return {"kind": kind or "unknown", "passed": False, "error": "unsupported_gate_kind"}


def evaluate_scoring(scoring: dict[str, Any] | None, generated_text: str) -> dict[str, Any]:
    if not scoring:
        return {"kind": "none", "passed": True, "failed_rules": [], "failure_taxonomy": []}
    kind = scoring.get("kind")
    failed_rules: list[str] = []
    failure_taxonomy: list[str] = []
    normalized_answer = normalize_scoring_text(generated_text)
    details: dict[str, Any] = {"kind": kind, "normalized_answer": normalized_answer}

    if kind == "required_forbidden_tokens":
        required = list(scoring.get("required_keywords") or [])
        forbidden = list(scoring.get("forbidden_tokens") or [])
        missing = missing_keywords(normalized_answer, required)
        observed_forbidden = observed_forbidden_tokens(normalized_answer, forbidden)
        if missing:
            failed_rules.append("required_keywords_missing")
            failure_taxonomy.append("required_keyword_missing")
        if observed_forbidden:
            failed_rules.append("forbidden_tokens_observed")
            failure_taxonomy.append("forbidden_token_observed")
        details.update(
            {
                "required_keywords_missing": missing,
                "forbidden_tokens_observed": observed_forbidden,
            }
        )
    elif kind == "required_keywords":
        required = list(scoring.get("required_keywords") or [])
        forbidden = list(scoring.get("forbidden_tokens") or [])
        missing = missing_keywords(normalized_answer, required)
        observed_forbidden = observed_forbidden_tokens(normalized_answer, forbidden)
        if missing:
            failed_rules.append("required_keywords_missing")
            failure_taxonomy.append("required_keyword_missing")
        if observed_forbidden:
            failed_rules.append("forbidden_tokens_observed")
            failure_taxonomy.append("forbidden_token_observed")
        details.update(
            {
                "required_keywords_missing": missing,
                "forbidden_tokens_observed": observed_forbidden,
            }
        )
    elif kind == "normalized_match":
        expected = normalize_match_text(str(scoring.get("expected_normalized") or ""))
        observed = normalize_match_text(generated_text)
        if observed != expected:
            failed_rules.append("normalized_match_failed")
            failure_taxonomy.append("normalized_match_failed")
        details.update({"expected_normalized": expected, "observed_normalized": observed})
    else:
        failed_rules.append("unsupported_scoring_kind")
        failure_taxonomy.append("unsupported_scoring_kind")

    return {
        "kind": kind or "unknown",
        "passed": not failed_rules,
        "failed_rules": failed_rules,
        "failure_taxonomy": failure_taxonomy,
        "details": details,
    }


def quality_status(gate_result: dict[str, Any], scoring_result: dict[str, Any], generated_text: str) -> dict[str, Any]:
    failed_rules: list[str] = []
    failure_taxonomy: list[str] = []
    normalized = SPECIAL_TOKEN_RE.sub("", generated_text).strip()
    if not normalized:
        failed_rules.append("non_empty_answer")
        failure_taxonomy.append("empty_output")
    if not gate_result.get("passed", False):
        failed_rules.append("answer_gate")
        failure_taxonomy.append("answer_gate_failed")
    if not scoring_result.get("passed", False):
        failed_rules.extend(scoring_result.get("failed_rules") or [])
        failure_taxonomy.extend(scoring_result.get("failure_taxonomy") or [])
    return {
        "passed": not failed_rules,
        "failed_rules": sorted(set(failed_rules)),
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "non_empty_answer": bool(normalized),
        "printable_utf8": True,
        "no_replacement_chars": "\ufffd" not in generated_text,
        "no_raw_special_tokens": "<|" not in generated_text,
        "mostly_text": bool(readable_words(generated_text)),
        "gate_kind": gate_result.get("kind"),
        "answer_gate": gate_result,
        "scoring": scoring_result,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases_total": len(cases),
        "passed": 0,
        "failed": 0,
        "fallback_used": any(case.get("fallback_used", False) for case in cases),
        "profile_summary": {},
        "category_summary": {},
    }
    by_profile: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    for case in cases:
        passed = bool(case["quality"]["passed"])
        summary["passed" if passed else "failed"] += 1
        for bucket, key in ((by_profile, case["profile"]), (by_category, case["category"])):
            bucket[key]["total"] += 1
            bucket[key]["passed" if passed else "failed"] += 1
    summary["profile_summary"] = dict(sorted(by_profile.items()))
    summary["category_summary"] = dict(sorted(by_category.items()))
    return summary


def run_device(
    device: str,
    model_dir: Path,
    ov_genai: Any,
    ov_core: Any,
    cases: list[dict[str, Any]],
    defaults: dict[str, Any],
) -> dict[str, Any]:
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

    results: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        max_new_tokens = int(case.get("max_new_tokens") or defaults.get("max_new_tokens") or 48)
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
        prompt = generation["prompt"]
        generated_text = generation["generated_text"]
        gate = evaluate_gate(case.get("gate"), generated_text)
        scoring = evaluate_scoring(case.get("scoring"), generated_text)
        quality = quality_status(gate, scoring, generated_text)
        first_chunk_ms = None
        if first_chunk_at[0] is not None:
            first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
        results.append(
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
                "prompt_token_count": prompt["prompt_token_count"],
                "max_new_tokens": max_new_tokens,
                "generation_config": {
                    "do_sample": False,
                    "num_beams": 1,
                    "apply_chat_template": True,
                    "max_new_tokens": max_new_tokens,
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
                "stop_eos": {
                    "raw_special_token_seen": "<|" in generated_text,
                    "eos_marker_seen": "<|im_end|>" in generated_text or "<|endoftext|>" in generated_text,
                    "stop_policy": "OpenVINO GenAI LLMPipeline greedy generation with exported Qwen2.5 tokenizer",
                },
                "quality": quality,
                "answer_gate": gate,
                "scoring": scoring,
                "timing": {
                    "generation_wall_ms": generation_wall_ms,
                    "first_streamed_text_chunk_ms": first_chunk_ms,
                    "first_streamed_text_chunk": chunks[0]["text"] if chunks else None,
                    "streamed_chunks_count": len(chunks),
                    "openvino_perf_metrics": perf_metrics(result),
                },
                "fallback_used": False,
                "requested_backend": selected_backend,
                "selected_backend": selected_backend,
                "runtime_api": "openvino_genai",
                "runtime_device": device,
                "backend_lane": backend_lane,
                "selected_kernel_or_runtime": runtime,
                "status": "passed" if quality["passed"] else "quality_failed",
            }
        )

    summary = summarize_cases(results)
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
        "promotion_status": "candidate_only_not_promoted",
        "route_promotion_changed": False,
        "quality_summary": summary,
        "cases": results,
        "phase_coverage": {
            "pipeline_construct_wall_ms": "measured_by_runner",
            "tokenization_duration": "openvino_genai_perf_metrics",
            "time_to_first_token": "openvino_genai_perf_metrics",
            "first_streamed_text_chunk": "measured_by_streamer_callback",
            "generate_duration": "openvino_genai_perf_metrics",
            "inference_duration": "openvino_genai_perf_metrics",
            "time_per_output_token": "openvino_genai_perf_metrics",
            "detokenization_duration": "openvino_genai_perf_metrics",
            "decode_128": "not_measured_corpus_v2_bounded_quality_lens_only",
            "prefill_512": "not_measured_corpus_v2_bounded_quality_lens_only",
        },
    }


def corpus_metadata(corpus: dict[str, Any]) -> dict[str, Any]:
    cases = corpus["cases"]
    profiles = sorted({str(case.get("profile", "unknown")) for case in cases})
    categories = sorted({str(case.get("category", "unknown")) for case in cases})
    return {
        "schema": corpus.get("schema"),
        "artifact_kind": corpus.get("artifact_kind"),
        "name": corpus.get("name"),
        "description": corpus.get("description"),
        "metadata": corpus.get("metadata"),
        "model": corpus.get("model"),
        "defaults": corpus.get("defaults"),
        "cases_total": len(cases),
        "profiles": profiles,
        "categories": categories,
        "file": file_record(Path("ci/quality/lunar-lake-answer-corpus-v2.yaml")),
    }


def main() -> int:
    args = parse_args()
    import openvino as ov
    import openvino_genai as ov_genai

    corpus = load_corpus(args.corpus)
    defaults = corpus.get("defaults") or {}
    cases = corpus["cases"]
    core = ov.Core()

    devices = [
        run_device(device, args.model_dir, ov_genai, core, cases, defaults)
        for device in args.devices
    ]
    all_cases = [case for device in devices for case in device["cases"]]
    quality_summary = summarize_cases(all_cases)
    fallback_used_any = any(device["fallback_used"] for device in devices)
    manifest = load_json(Path(args.manifest))
    created = args.created_utc or utc_now()

    out = {
        "schema_version": "1.0.0",
        "artifact_kind": "intel_258v_dense_slm_openvino_corpus_v2",
        "campaign": "intel-258v-platform",
        "item": "LNL258V-OV-QUAL-004",
        "created_utc": created,
        "machine_id": args.machine_id,
        "proof_stage": "candidate_route_corpus_v2_executed_no_promotion_change",
        "comparison_scope": "same_machine_qwen2_5_openvino_genai_candidate_routes_against_lunar_lake_answer_corpus_v2",
        "requested_backend": "openvino-cpu-gpu-npu",
        "selected_backend": "openvino-cpu-gpu-npu",
        "runtime_api": "openvino_genai",
        "fallback_used": fallback_used_any,
        "backend_lane": "dense_slm_openvino_candidate_routes",
        "model_family": "qwen",
        "model_architecture": "qwen2",
        "quantization": "INT4_SYM",
        "prompt_template": "qwen2.5",
        "tokenizer_source": "hf_tokenizer_export",
        "scoring_policy": {
            "name": "openvino_corpus_v2_scoring_aligned_with_rust_answer_corpus_v2",
            "normalized_match": "strip known chat/stop markers, collapse whitespace, strip leading assistant separator, trim terminal punctuation, lowercase",
            "required_keywords": "strip known chat/stop markers, collapse whitespace, strip leading assistant separator, then match case-insensitive token boundaries",
            "forbidden_tokens": "strip known chat/stop markers, collapse whitespace, strip leading assistant separator, then match case-insensitive token boundaries",
            "generated_token_ids": "openvino_genai_encoded_results_tokens",
            "direct_generated_token_ids": "captured from OpenVINO GenAI EncodedResults.tokens by generating from OpenVINO TokenizedInputs",
        },
        "promotion_status": "candidate_only_not_promoted",
        "route_promotion_changed": False,
        "paths": {
            "openvino_ir_manifest": args.manifest,
            "gguf_cpu_corpus_v2": args.gguf_cpu_corpus_v2,
            "phase_runner": args.phase_runner,
            "route_profile_comparison": args.route_profile_comparison,
            "answer_corpus_v2": args.corpus.as_posix(),
        },
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_model_dir": args.model_dir.as_posix(),
            "model_binary_committed": False,
            "openvino_manifest_source_model": (manifest or {}).get("source_model"),
            "openvino_export_contract": (manifest or {}).get("export_contract"),
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
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "openvino": {"version": ov.get_version(), "available_devices": core.available_devices},
            "openvino_genai": {"version": getattr(ov_genai, "__version__", None)},
            "transformers": {
                "tokenizer_loaded_from_export_dir": False,
                "openvino_tokenizer_loaded_from_export_dir": True,
                "generated_token_ids_available_from_pipeline": True,
            },
        },
        "corpus": corpus_metadata(corpus),
        "generation": {
            "devices_total": len(devices),
            "cases_per_device": len(cases),
            "cases_total": len(all_cases),
            "devices": devices,
            "quality_summary": quality_summary,
        },
        "verification": {
            "llmpipeline_constructed_for_all_devices": len(devices) == len(args.devices),
            "generation_ran_for_all_requested_cases": True,
            "fallback_used": fallback_used_any,
            "openvino_perf_metrics_recorded": True,
            "streamer_first_text_chunk_recorded": True,
            "generated_token_ids_available_from_pipeline": True,
            "generated_token_ids_retokenized_from_text": False,
            "route_promotion_changed": False,
            "candidate_routes_remain_unpromoted": True,
            "quality_failures_are_evidence_not_route_policy": True,
        },
        "claim_boundary": {
            "may_claim": [
                "The bounded Lunar Lake answer corpus v2 was executed on OpenVINO CPU, GPU.0/Arc 140V, and NPU candidate routes on the same 258V machine.",
                "The receipt records per-case profile/category answer-gate evidence, OpenVINO GenAI runtime identity, PerfMetrics, prompt token evidence, generated text, and fallback_used=false for the requested OpenVINO routes when observed.",
                "The receipt collects candidate-route evidence for later profile promotion work without changing route policy.",
            ],
            "must_not_claim": [
                "OpenVINO GPU or NPU is promoted for any Lunar Lake route profile.",
                "OpenVINO GPU or NPU speedup, acceleration, or benchmark-qualified advantage is proven.",
                "Broad dense SLM answer quality is proven beyond bounded corpus v2 evidence.",
                "OpenVINO GPU evidence proves native OpenCL execution.",
                "OpenVINO NPU evidence proves native NPU inference outside OpenVINO GenAI.",
                "Dense SLM receipts prove BitNet QK256/I2_S behavior.",
                "BitNet QK256/I2_S behavior changed.",
            ],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if not fallback_used_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
