# SLM CPU Campaign

Campaign ID: `slm-cpu`

Status: active

## Objective

Make the Intel i5-8250U a strict CPU proof host for small dense transformer GGUF models without reusing BitNet QK256/I2_S assumptions or claiming useful throughput.

## End State

- A real dense GGUF candidate is selected by verified metadata, not model-name recognition.
- Tokenizer resolution is strict and receipt-backed.
- Dense architecture metadata is normalized into an adapter contract.
- Tiny deterministic answer receipts record prompt IDs, generated IDs, decoded text, backend/kernel identity, and fallback state.
- Failures are diagnosable before any answer-quality or performance claim is made.

## Hard Constraints

- Do not edit BitNet QK256/I2_S kernels.
- Do not change BitNet proof receipt semantics.
- Do not claim sustained 8250U throughput.
- Do not claim server, GPU, OpenVINO, UHD 620, or NPU execution.
- Do not treat model names as proof of GGUF architecture/tokenizer compatibility.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| SLM-CPU-000 | merged | 8250U dense SLM CPU lane merged in #3902. |
| SLM-CPU-001 | merged | Model candidate manifest, artifact policy, and 8250U runbook merged in #3905. |
| SLM-CPU-002 | merged | Dense Qwen preflight foundations merged in #3910. |
| SLM-CPU-002A | merged | Dense Qwen strict-load blocker hardening merged in #3917. |
| SLM-CPU-002B | merged | Dense standard GGUF Q8_0/Q*_K support merged in #3926. |
| SLM-CPU-003 | merged | Tiny strict dense CPU receipt merged in #3940. |
| SLM-CPU-004 | merged | SLM answer corpus evidence merged in #3957. |
| SLM-CPU-005 | merged | Reference divergence artifact schema and validator merged in #3969. |
| SLM-CPU-006 | pr_open (#4031) | Capture Qwen3-0.6B first-token divergence with bitnet-rs and known-good reference top-k/logit evidence. |

## Review Policy

SLM CPU PRs must stay separate from BitNet CPU proof PRs. Dense transformer adapter work may reuse loader, tokenizer, and receipt infrastructure, but must not reuse QK256/I2_S layout assumptions or modify accelerator/server lanes.
