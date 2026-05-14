#!/usr/bin/env python3
"""Capture direct BitNet.cpp generated-token and top-k evidence.

This helper drives a locally built BitNet.cpp `llama-server` for the fixed
Lunar Lake BitNet reference prompts. The stock server response omits token IDs
and logits from `completion_probabilities`; apply
`ci/bitnet_cpp_server_token_logits.patch` to the BitNet.cpp checkout before
building when direct IDs/logits are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE = Path("ci/hardware/intel-258v/2026-05-08/hf-prompt-token-reference-parity-after-prompt-fix.json")
DEFAULT_TEXT_BOUNDARY = Path("ci/hardware/intel-258v/2026-05-08/external-first-token-reference.json")
DEFAULT_OUTPUT = Path("ci/hardware/intel-258v/2026-05-08/external-first-token-reference-direct.json")
DEFAULT_MODEL = Path("C:/Code/Models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
DEFAULT_SERVER = Path.home() / ".cache/bitnet_cpp/build/bin/llama-server.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--text-boundary-reference", type=Path, default=DEFAULT_TEXT_BOUNDARY)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--n-probs", type=int, default=5)
    parser.add_argument("--created-utc")
    parser.add_argument("--server-already-running", action="store_true")
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


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(base_url: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            get_json(f"{base_url}/health", timeout=2)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise SystemExit(f"BitNet.cpp server did not become ready: {last_error}")


def start_server(args: argparse.Namespace, base_url: str) -> subprocess.Popen[str] | None:
    if args.server_already_running:
        wait_for_health(base_url)
        return None

    if not args.server_bin.is_file():
        raise SystemExit(f"server binary not found: {args.server_bin}")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    log_dir = Path(tempfile.gettempdir())
    stdout_path = log_dir / "bitnet_cpp_reference_server.stdout.log"
    stderr_path = log_dir / "bitnet_cpp_reference_server.stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr = stderr_path.open("w", encoding="utf-8", errors="replace")
    cmd = [
        str(args.server_bin),
        "-m",
        str(args.model),
        "--override-kv",
        "tokenizer.ggml.pre=str:llama-bpe",
        "--no-mmap",
        "-ngl",
        "0",
        "-t",
        str(args.threads),
        "-c",
        str(args.ctx_size),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    try:
        wait_for_health(base_url)
    except BaseException:
        proc.terminate()
        raise
    return proc


def token_ids_from_tokenize_response(response: dict[str, Any]) -> list[int]:
    tokens = response.get("tokens", [])
    result: list[int] = []
    for token in tokens:
        if isinstance(token, int):
            result.append(token)
        elif isinstance(token, dict) and isinstance(token.get("id"), int):
            result.append(token["id"])
    return result


def first_token_topk(item: dict[str, Any]) -> list[dict[str, Any]]:
    topk = item.get("probs")
    if not isinstance(topk, list):
        return []
    normalized: list[dict[str, Any]] = []
    for candidate in topk:
        if not isinstance(candidate, dict):
            continue
        normalized.append(
            {
                "token_id": candidate.get("tok_id"),
                "token_text": candidate.get("tok_str"),
                "logit": candidate.get("logit"),
                "probability": candidate.get("prob"),
            }
        )
    return normalized


def normalized_cases(source: dict[str, Any], text_boundary: dict[str, Any]) -> list[dict[str, Any]]:
    text_cases = {case.get("case_id"): case for case in text_boundary.get("cases", [])}
    result: list[dict[str, Any]] = []
    for case in source.get("cases", []):
        reference = case.get("reference", {})
        case_id = case.get("case_id")
        text_case = text_cases.get(case_id, {})
        rendered_prompt = case.get("reference_prompt") or reference.get("rendered_prompt")
        if not rendered_prompt:
            raise SystemExit(f"case `{case_id}` does not expose a rendered reference prompt")
        result.append(
            {
                "case_id": case_id,
                "question": case.get("question") or case.get("prompt") or text_case.get("question"),
                "reference_prompt": rendered_prompt,
                "reference_prompt_mode": case.get("reference_prompt_mode")
                or "hf_apply_chat_template_after_prompt_fix",
                "prompt_token_ids": case.get("prompt_token_ids") or reference.get("prompt_token_ids", []),
                "max_new_tokens": text_case.get("max_new_tokens", 4),
                "reference_generated_text": text_case.get("reference_generated_text"),
            }
        )
    return result


def capture_case(base_url: str, case: dict[str, Any], n_probs: int) -> dict[str, Any]:
    prompt = str(case["reference_prompt"])
    tokenized = post_json(
        f"{base_url}/tokenize",
        {"content": prompt, "add_special": True, "with_pieces": True},
        timeout=30,
    )
    prompt_token_ids = token_ids_from_tokenize_response(tokenized)

    response = post_json(
        f"{base_url}/completion",
        {
            "prompt": prompt,
            "n_predict": int(case.get("max_new_tokens", 4)),
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "seed": 42,
            "n_probs": n_probs,
            "stream": False,
            "cache_prompt": False,
        },
        timeout=180,
    )
    probabilities = response.get("completion_probabilities", [])
    if not isinstance(probabilities, list):
        probabilities = []

    missing_fields: list[str] = []
    if not probabilities:
        missing_fields.extend(["generated_token_ids", "first_token_top_k_logits"])
    elif not all(isinstance(item, dict) and isinstance(item.get("tok_id"), int) for item in probabilities):
        missing_fields.append("generated_token_ids")
    if probabilities and not all(
        isinstance(item, dict)
        and isinstance(item.get("probs"), list)
        and all(isinstance(candidate, dict) and "logit" in candidate and "tok_id" in candidate for candidate in item["probs"])
        for item in probabilities
    ):
        missing_fields.append("first_token_top_k_logits")

    generated_token_ids = [
        int(item["tok_id"])
        for item in probabilities
        if isinstance(item, dict) and isinstance(item.get("tok_id"), int)
    ]
    first_generated_token_id = generated_token_ids[0] if generated_token_ids else None
    decoded_first_token = probabilities[0].get("content") if probabilities and isinstance(probabilities[0], dict) else None

    first_topk = first_token_topk(probabilities[0]) if probabilities and isinstance(probabilities[0], dict) else []
    generated_token_records = [
        {
            "token_id": item.get("tok_id"),
            "token_text": item.get("content"),
            "top_k": first_token_topk(item),
        }
        for item in probabilities
        if isinstance(item, dict)
    ]

    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "reference_prompt": prompt,
        "reference_prompt_mode": case.get("reference_prompt_mode"),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_source": "BitNet.cpp llama-server /tokenize add_special=true with tokenizer.ggml.pre=str:llama-bpe",
        "expected_prompt_token_ids_without_local_bos": case.get("prompt_token_ids", []),
        "max_new_tokens": int(case.get("max_new_tokens", 4)),
        "reference_generated_text": response.get("content", ""),
        "expected_text_boundary_from_previous_reference": case.get("reference_generated_text"),
        "first_generated_token_id": first_generated_token_id,
        "decoded_first_token": decoded_first_token,
        "generated_token_ids": generated_token_ids,
        "generated_token_records": generated_token_records,
        "generated_token_ids_available": bool(generated_token_ids),
        "logits_available": bool(first_topk) and all("logit" in item and item.get("logit") is not None for item in first_topk),
        "first_token_top_k_logits": first_topk,
        "server_response_summary": {
            "tokens_predicted": response.get("tokens_predicted"),
            "tokens_evaluated": response.get("tokens_evaluated"),
            "stopped_eos": response.get("stopped_eos"),
            "stopped_limit": response.get("stopped_limit"),
            "timings": response.get("timings"),
        },
        "missing_reference_fields": missing_fields,
    }


def build_receipt(
    args: argparse.Namespace,
    source: dict[str, Any],
    text_boundary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    cases_with_ids = sum(1 for case in cases if case["generated_token_ids_available"])
    cases_with_logits = sum(1 for case in cases if case["logits_available"])
    created = args.created_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_external_reference_direct_token_boundary",
        "item": "CPU-BITNET-REF-003",
        "machine_id": "intel-258v",
        "proof_stage": "external_reference_direct_tokens_recorded",
        "created_utc": created,
        "source_reference": args.source_reference.as_posix(),
        "text_boundary_reference": args.text_boundary_reference.as_posix(),
        "reference": {
            "runner": "Microsoft BitNet.cpp / llama-server",
            "bitnet_cpp_commit": text_boundary["reference"].get("bitnet_cpp_commit"),
            "llama_cpp_submodule_commit": text_boundary["reference"].get("llama_cpp_submodule_commit"),
            "llama_cli_version": text_boundary["reference"].get("llama_cli_version"),
            "server_binary": str(args.server_bin),
            "server_patch": "ci/bitnet_cpp_server_token_logits.patch",
            "server_patch_status": "required_for_tok_id_and_logit_fields",
            "command_shape": "llama-server.exe -m <ggml-model-i2_s.gguf> --override-kv tokenizer.ggml.pre=str:llama-bpe --no-mmap -ngl 0 -t 8 -c 4096 --host 127.0.0.1 --port <port>",
            "completion_request": {
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "min_p": 0,
                "seed": 42,
                "n_probs": args.n_probs,
                "stream": False,
                "cache_prompt": False,
            },
            "generated_token_ids_available": cases_with_ids == len(cases),
            "first_generated_token_id_available": cases_with_ids == len(cases),
            "first_token_topk_logits_available": cases_with_logits == len(cases),
            "logits_available": cases_with_logits == len(cases),
        },
        "model": {
            **text_boundary.get("model", {}),
            "local_path": str(args.model),
            "local_sha256": sha256_file(args.model),
        },
        "tokenizer": text_boundary.get("tokenizer", {}),
        "summary": {
            "cases_total": len(cases),
            "cases_with_reference_generated_token_ids": cases_with_ids,
            "cases_with_reference_first_token_topk_logits": cases_with_logits,
            "reference_generated_token_ids_available": cases_with_ids == len(cases),
            "reference_logits_available": cases_with_logits == len(cases),
            "boundary_classification": "direct_reference_generated_token_ids_and_first_token_topk_logits_recorded"
            if cases_with_ids == len(cases) and cases_with_logits == len(cases)
            else "direct_reference_evidence_incomplete",
            "next_required_evidence": "rerun first-token-divergence against this direct reference artifact",
        },
        "cases": cases,
        "fallback_used": False,
        "claim_boundary": {
            "may_claim": [
                "BitNet.cpp direct generated-token IDs are recorded for the fixed 258V reference prompts.",
                "BitNet.cpp first-token top-k token IDs, probabilities, and raw candidate logits are recorded from the patched llama-server response.",
                "The evidence is suitable for rerunning the first-token divergence classifier against local scalar/AVX2 receipts.",
            ],
            "must_not_claim": [
                "BitNet-rs generated-token-ID parity against BitNet.cpp is proven by this receipt alone.",
                "BitNet-rs first-token logits parity against BitNet.cpp is proven by this receipt alone.",
                "Broad BitNet answer quality is proven.",
                "CPU speedup or sustained performance is proven.",
                "Arc 140V execution or acceleration is involved.",
                "Intel NPU execution or acceleration is involved.",
                "QK256/I2_S kernel behavior is changed by this receipt.",
                "Full model correctness is proven.",
            ],
        },
    }


def main() -> int:
    args = parse_args()
    source = read_json(args.source_reference)
    text_boundary = read_json(args.text_boundary_reference)
    base_url = f"http://{args.host}:{args.port}"
    proc = start_server(args, base_url)
    try:
        cases = [capture_case(base_url, case, args.n_probs) for case in normalized_cases(source, text_boundary)]
        receipt = build_receipt(args, source, text_boundary, cases)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
        print(f"BitNet.cpp direct reference boundary written to {args.json_out}")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
