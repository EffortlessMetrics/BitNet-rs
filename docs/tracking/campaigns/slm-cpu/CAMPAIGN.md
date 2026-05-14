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
| SLM-CPU-006A | merged | First-token divergence classifier, logits alias, and 8250U workflow support merged in #4051. |
| SLM-CPU-006B | merged | Qwen3-0.6B first-token divergence artifact capture merged in #4096. |
| SLM-CPU-007A | merged | Qwen3 bounded checkpoint probe support merged in #4132. |
| SLM-CPU-007B | merged | First comparable drift capture merged in #4138. |
| SLM-CPU-008 | merged | Qwen3 architecture-default parity candidate merged in #4434. |
| SLM-CPU-008R | merged | Post-#4434 real artifact revalidation merged in #4572; first-token parity remained unproven. |
| SLM-CPU-008S | merged | Output-head/shared-math root-cause localization merged in #4606. |
| SLM-CPU-008T | merged | Dedicated output.weight selection candidate merged in #4611. |
| SLM-CPU-008U | merged | GGUF output.weight layout candidate merged in #4617. |
| SLM-CPU-008V | merged | Post-008U real i5-8250U artifact refresh merged in #4633; first-token parity remained unproven and the official GGUF uses tied token embeddings. |
| SLM-CPU-008W | merged | Tied-token-embedding logits audit merged in #4641; output-head/vocab boundary remains insufficient to prove first-token parity. |
| SLM-CPU-008X | merged | Checkpoint-aware reference comparison support merged in #4655; real known-good checkpoint capture remains separate. |
| SLM-CPU-008YA | merged | Qwen no-thinking prompt control merged in #4669; first-token parity was not claimed. |
| SLM-CPU-008YB | merged | Qwen thinking special-token preservation merged in #4699; refreshed no-thinking reference prompt-ID comparison no longer literalizes `<think>` markers. |
| SLM-CPU-008Y1 | merged | Reference checkpoint capture method merged in #4769; interactive llama.cpp output and top-k-only evidence do not satisfy checkpoint-pack acceptance. |
| SLM-CPU-008Y | ready | Capture or ingest the real Qwen3 reference checkpoint pack now that the capture method is documented; do not claim first-token parity until checkpoint-aware validation identifies the first drift. |

## Review Policy

SLM CPU PRs must stay separate from BitNet CPU proof PRs. Dense transformer adapter work may reuse loader, tokenizer, and receipt infrastructure, but must not reuse QK256/I2_S layout assumptions or modify accelerator/server lanes.
