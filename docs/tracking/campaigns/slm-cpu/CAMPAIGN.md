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
| SLM-CPU-000 | pr_open | Define the 8250U dense SLM CPU lane and first-target policy in #3902. |
| SLM-CPU-001 | ready | Add model candidate manifest, artifact policy, and 8250U runbook. |
| SLM-CPU-002 | proposed | Add strict dense GGUF metadata preflight. |
| SLM-CPU-003 | proposed | Run the first tiny dense CPU receipt. |
| SLM-CPU-004 | proposed | Add SLM answer corpus evidence. |
| SLM-CPU-005 | proposed | Add reference divergence artifact schema. |

## Review Policy

SLM CPU PRs must stay separate from BitNet CPU proof PRs. Dense transformer adapter work may reuse loader, tokenizer, and receipt infrastructure, but must not reuse QK256/I2_S layout assumptions or modify accelerator/server lanes.
