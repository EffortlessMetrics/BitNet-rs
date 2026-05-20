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
| SLM-CPU-020 | merged | #5108 retries SmolLM2 strict CPU sanity after exact metadata-scoped normalization validation; it reaches tokenizer loading, prompt rendering, and one-token generation with `fallback_used=false`, but the math prompt generates `The`, so CPU answer readiness remains false and the next proof is wrong-first-token diagnosis before CUDA planning. |
| SLM-CPU-021 | merged | #5133 merged the SmolLM2 wrong-first-token diagnosis from committed reference-runner, prompt/tokenizer, and strict CPU retry evidence; CPU answer readiness and CUDA planning remain blocked until a reference-compatible first-token/top-k or checkpoint comparator localizes the fault. |
| SLM-CPU-022 | merged | #5140 merged the SmolLM2 comparator contract and `reference-compare` fixture coverage for the first-token/top-k artifact shape. This is support only; a fresh same-prompt external reference comparator remains required before CPU answer readiness or CUDA planning. |
| SLM-CPU-023 | merged | #5312 formalized the Kaby Lake Qwen3 Q8_0 performance-dashboard baseline from existing 1/2/4/8-thread envelope and operator-profile receipts; this is a baseline/decision surface only, not a new runtime optimization or sustained-throughput claim. |
| SLM-CPU-024 | merged | #5342 added the guarded greedy no-penalty sampler fast path while preserving the Qwen3 Q8_0 4-thread warm-session behavior oracle. |
| SLM-CPU-025 | merged | #5357 isolated deterministic greedy logits extraction so exact no-penalty steps use direct tensor argmax; default repetition-penalty vector extraction remained explicit. |
| SLM-CPU-026 | merged | #5369 reused a host logits scratch buffer for default repetition-penalty decode steps, reducing fresh logits Vec extraction while preserving generated IDs, strict tokenizer authority, cpu-rust, and fallback=false. |
| SLM-CPU-027 | blocked | Duplicate second-model item blocked because SLM-CPU-017 and SLM-CPU-017A already recorded the SmolLM2 blocker and positive Qwen2.5 Q8_0 second-model sanity evidence. |
| SLM-CPU-028 | merged | #5384 defined the bounded Q4_K_M/Q4_K_S expansion plan: candidate artifact identity, metadata, tokenizer, strict CPU, fallback, corpus, multi-token, warm-session, and operator-profile gates before any Q4 runtime support claim. |
| SLM-CPU-029 | merged | #5405 reduced the next Qwen3 Q8_0 warm-session allocation/layout boundary while preserving the 4-thread generated-ID oracle and strict provenance. |
| SLM-CPU-030 | merged | #5457 added prompt setup allocation attribution for buffer reset, token seed, KV-cache, and sampler setup subcomponents. |
| SLM-CPU-031 | merged | #5499 reused a single CPU KV cache across warm-session prompts while preserving prompt isolation through explicit clears. |
| SLM-CPU-032 | merged | #5514 reused rendered prompt token IDs across repeated warm-session prompts and recorded prompt-token cache hit/miss counts. |
| SLM-CPU-033 | merged | #5603 records the dominant aggregate allocation hotspot and next evidence-scoped optimization target after prompt-token caching; this is diagnostic prioritization only, not a speedup or sustained-throughput claim. |
| SLM-CPU-034 | merged | #5612 attributes the dominant `prompt_prefill` / `model.forward` allocation boundary before changing dense math. |
| SLM-CPU-035 | merged | Regenerate the real i5-8250U Qwen3 Q8_0 warm-session artifact so the receipt records `prompt_prefill_breakdown.embed` and `prompt_prefill_breakdown.forward`. |
| SLM-CPU-036 | merged | #5625 classifies `prompt_prefill.forward` as transformer forward workspace and owned tensor outputs before changing dense math. |
| SLM-CPU-037 | merged | #5643 explicitly rules out caller-side transformer-forward buffer reuse at the current owned tensor-output boundary and points the next safe hook at a typed transformer forward workspace API. |
| SLM-CPU-038 | merged | #5679 introduced the first typed transformer forward workspace API boundary while preserving the existing owned tensor-output behavior oracle. |
| SLM-CPU-039 | merged | #5693 routed the feed-forward output through the typed workspace and recorded `feed_forward.output` as the first workspace-owned transformer output boundary while reusable Candle storage remains deferred. |
| SLM-CPU-040 | merged | #5715 classified the exact `FeedForward::down_proj` output boundary and recorded that reusable workspace-backed storage remains blocked by Candle's owned linear output path. |
| SLM-CPU-041 | merged | #5754 narrowed the `FeedForward::down_proj` storage blocker to the exact Candle tensor API gap: linear weight/bias are readable, but matmul/bias-add still lack caller-provided output storage. |
| SLM-CPU-042 | merged | #5773 identifies and instruments the first behavior-preserving Q8_0 dense linear locality/dequant boundary after the Candle output-storage API blocker. |
| SLM-CPU-043 | merged | #5794 added a fixture-level packed Q8_0 sidecar linear prototype that matches eager F32 fixture output without replacing production runtime compute. |
| SLM-CPU-044 | merged | #5810 defined the first production-integration boundary for carrying packed Q8_0 sidecar metadata toward runtime dense-linear use while keeping eager F32 Candle tensors as the behavior oracle. |
| SLM-CPU-045 | merged | #5845 preserves packed Q8_0 sidecar metadata from strict GGUF tensor loading into an inert model-side descriptor without changing generation behavior or claiming speedup. |
| SLM-CPU-046 | merged | #5860 added a dense-linear dispatch selector that keeps eager F32 Candle selected while exposing packed Q8_0 sidecars only as unavailable candidates. |
| SLM-CPU-047 | merged | #5868 added the packed Q8_0 sidecar equivalence gate that records fixture parity and keeps runtime compute disabled until generated-ID/text receipt equivalence exists. |
| SLM-CPU-048 | merged | #5873 added the non-executing packed Q8_0 sidecar runtime preflight that names generated-ID receipt, production compute hook, and eager-selector blockers. |
| SLM-CPU-049 | merged | Added the generated-ID/text receipt equivalence gate before any packed Q8_0 sidecar runtime selection or speedup claim. |
| SLM-CPU-050 | merged | Added the production-compute-hook availability surface while keeping eager F32 Candle selected until later behavior-preserving selector evidence exists. |
| SLM-CPU-051 | merged | Add the selector-readiness gate that names the evidence required before a later packed Q8_0 runtime selector update. |
| SLM-CPU-052 | merged | #5921 implemented the first behavior-preserving selector update while preserving eager F32 Candle as the runtime oracle unless generated-ID/text evidence allows a packed Q8_0 candidate. |
| SLM-CPU-053 | merged | #5943 validated the first packed Q8_0 sidecar runtime execution proof gate and recorded that production runtime execution remains blocked by eager F32 dispatch, disabled packed runtime compute, missing production runtime hook, and missing before/after receipts. |
| SLM-CPU-054 | merged | #5992 recorded the remaining packed Q8_0 sidecar runtime hook/API gap while keeping eager F32 Candle as the default behavior oracle and sidecar_runtime_compute_allowed=false. |
| SLM-CPU-055 | merged | #6008 added the first production dense-linear hook contract gate so transformer dense linear calls can receive an explicit eager-F32 selection or selected Q8_0 sidecar descriptor while keeping packed compute disabled until before/after behavior receipts exist. |
| SLM-CPU-056 | merged | #6018 implemented the first production dense-linear hook boundary from SLM-CPU-055 without enabling packed Q8_0 sidecar compute by default or weakening the Qwen3 Q8_0 Kaby behavior oracle. |
| SLM-CPU-057 | merged | #6045 added the next Qwen3 Q8_0 dense hook-selection receipt gate before any packed Q8_0 sidecar compute can be enabled. |
| SLM-CPU-058 | merged | #6070 queued the dense-hook before/after receipt gate and preserved the current safe state: packed Q8_0 sidecar compute remains disabled until behavior-preserving before/after receipts and a narrow compute proof exist. |
| SLM-CPU-059 | merged | #6078 recorded the packed Q8_0 compute-kernel proof gate and blocker artifact: production transformer dense-linear hooks still receive metadata-only sidecar descriptors, so payload-bearing packed Q8_0 sidecar compute remains disabled pending before/after behavior receipts. |
| SLM-CPU-060 | ready | Next evidence-scoped hook-contract slice: add a payload-bearing packed Q8_0 sidecar hook contract for one Qwen3 Q8_0 dense-linear tensor path, or record a concrete blocker if the current API cannot safely carry packed payload bytes to the dense-linear callsite. |

## Review Policy

SLM CPU PRs must stay separate from BitNet CPU proof PRs. Dense transformer adapter work may reuse loader, tokenizer, and receipt infrastructure, but must not reuse QK256/I2_S layout assumptions or modify accelerator/server lanes.
