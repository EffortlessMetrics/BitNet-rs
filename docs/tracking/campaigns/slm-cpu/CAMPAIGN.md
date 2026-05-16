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
| SLM-CPU-008Y | merged | Reference checkpoint artifact merged in #4778; validation localizes the first required-stage drift at `attention.q_proj` without claiming first-token parity or answer quality. |
| SLM-CPU-008Z | merged | Qwen3 q_proj drift localization merged in #4789; the default-prompt first token matches the reference while checkpoint drift remains bounded to projection arithmetic evidence, without claiming no-thinking token-19 parity or answer quality. |
| SLM-CPU-008AA | merged | No-thinking first-token reference/checkpoint artifact merged in #4817; prompt-policy mismatch remains distinct from answer quality. |
| SLM-CPU-008AB | merged | Constrained-answer target calibration merged in #4822; the original no-thinking math seed is unsuitable because the reference also chooses `2`. |
| SLM-CPU-008AC | merged | Constrained-answer scoring policy merged in #4826; SLM-CPU-009 could begin from the selected bounded scoring policy. |
| SLM-CPU-009A | merged | Qwen no-thinking corpus runner support merged in #4832. |
| SLM-CPU-009B | merged | First full strict no-thinking tiny-corpus evidence merged in #4843; records 4/5 passing cases and preserves the math miss. |
| SLM-CPU-009 | merged | Calibrated strict Qwen3 tiny corpus green artifact merged in #4846; records 5/5 passing constrained cases without claiming broad answer quality. |
| SLM-CPU-010 | merged | Bounded deterministic multi-token decode stability evidence merged in #4851. |
| SLM-CPU-011 | merged | Bounded strict i5-8250U Qwen3 warm-session receipts landed in #4858. |
| SLM-CPU-012 | merged | Bounded Qwen3 Q8_0 warm-session allocation/KV/layout cleanup landed in #4876; generated IDs remain the behavior oracle. |
| SLM-CPU-013 | merged | Bounded Qwen3 Q8_0 dense linear no-bias hot-path cleanup landed in #4891; generated IDs and strict provenance remain the behavior oracle. |
| SLM-CPU-014 | merged | Bounded dense output-head zero-bias allocation cleanup landed in #4900; generated IDs and strict provenance remain the behavior oracle. |
| SLM-CPU-015 | merged | Bounded i5-8250U Qwen3 Q8_0 warm-session thread and timing envelope evidence landed in #4911; generated IDs and strict provenance remain the behavior oracle. |
| SLM-CPU-016 | merged | Kaby Lake Qwen3 Q8_0 operator appliance profile host-context support merged in #4922; receipts now record process memory and storage/free-space where available while preserving explicit unavailable thermal/power fields. |
| SLM-CPU-017 | merged | Bounded SmolLM2 360M Q8_0 strict CPU preflight blocker evidence landed in #5041; strict loading fails closed before tokenizer/prompt/generation, with no Q4/Q5, accelerator, server, Qwen3.5, throughput, or BitNet QK256 claim. |
| SLM-CPU-017A | merged | Positive Qwen2.5-0.5B Q8_0 second-model sanity evidence landed in #5060; strict CPU receipt records prompt/generated IDs, GGUF tokenizer authority, cpu-rust, dense-qwen-cpu-reference, and fallback=false without broad quality or throughput claims. |
| SLM-CPU-018 | merged | SmolLM2 360M normalization policy audit landed in #5067: generic `llama` strict LayerNorm gamma validation remains fail-closed, and any SmolLM2 loader exception must be governed by exact artifact/model-family metadata before CPU sanity can be retried. |
| SLM-CPU-019 | merged | Exact metadata-scoped SmolLM2 360M normalization validation landed in #5081: generic `llama` strict LayerNorm/RMSNorm gamma validation remains fail-closed, and the SmolLM2 exception requires exact artifact SHA, GGUF metadata, and dimensions before the next strict CPU sanity retry. No CPU answer, CUDA, throughput, server, broad dense GGUF, or BitNet QK256 claim is made. |
| SLM-CPU-020 | in progress | SmolLM2 strict CPU sanity retry reaches tokenizer loading, prompt rendering, and one-token generation with `fallback_used=false`, but the math prompt generates `The`; CPU answer readiness remains false and the next proof is wrong-first-token diagnosis before CUDA planning. |

## Review Policy

SLM CPU PRs must stay separate from BitNet CPU proof PRs. Dense transformer adapter work may reuse loader, tokenizer, and receipt infrastructure, but must not reuse QK256/I2_S layout assumptions or modify accelerator/server lanes.
