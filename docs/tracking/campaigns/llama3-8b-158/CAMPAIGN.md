# Llama3 8B 1.58 Candidate Campaign

Campaign ID: `llama3-8b-158`

Status: active

## Objective

Make `HF1BitLLM/Llama3-8B-1.58-100B-tokens` a first-class BitNet-rs large
supported-model candidate without overclaiming. The campaign proves artifact,
tokenizer, prompt, conversion/runner, route-layout, reference quality, CPU,
accelerator, benchmark, and product gates in order.

## End State

BitNet-rs can state, with receipts, that an exact HF1BitLLM revision and file
hash set, using an approved tokenizer/prompt authority and conversion/runner
route, produces bounded reference-good output, then matching Rust CPU output,
then optional backend output for exact routes and profiles with
`fallback=false`.

## Hard Constraints

- Do not commit model binaries.
- Do not inherit official Microsoft 2B I2_S/QK256 proof.
- Do not inherit dense Llama3 or dense Qwen proof.
- Do not claim answer readiness from safetensors inventory.
- Do not claim I2_S/QK256 compatibility before layout proof.
- Do not route TL1/TL2 through QK256/I2_S kernels.
- Do not claim CUDA, Apple, CPU, server, or speed readiness before exact
  receipts prove the exact route.

## Work Items

| Work item | Status | Notes |
| --- | --- | --- |
| LLAMA3-158-000 | ready | Add source-of-truth docs, conservative matrices, and spec contracts. |
| LLAMA3-158-001 | proposed | Record artifact inventory receipt. |
| LLAMA3-158-002 | proposed | Verify or block conversion and reference runner authority. |
| LLAMA3-158-003 | proposed | Audit tokenizer/prompt authority and reference-good corpus. |
| LLAMA3-158-004 | proposed | Prove I2_S structural load and scalar CPU oracle. |
| LLAMA3-158-005 | proposed | Add exact backend and performance receipts only after CPU answer proof. |

## Review Policy

Documentation/spec/config work for this campaign is non-stackable and must not
touch runtime kernels. Runtime/backend work is blocked until the earlier
artifact, conversion/runner, tokenizer/prompt, and reference-quality gates pass.
