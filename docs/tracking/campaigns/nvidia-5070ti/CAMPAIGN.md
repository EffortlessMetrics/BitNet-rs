# NVIDIA 5070 Ti Campaign

Campaign ID: `nvidia-5070ti`

Status: active

## Objective

Maintain the completed RTX 5070 Ti CUDA BitNet proof lane while qualifying
strict BitNet CUDA performance with selected-device receipts and no CPU,
OpenCL, WGPU, dense CUDA, or generic GPU conflation.

## End State

- RTX 5070 Ti CUDA backend identity is distinct from generic CUDA and WGPU.
- CUDA and NVML probe facts are recorded before kernel execution claims.
- CUDA smoke, parity, receipts, and benchmarks are sequenced after identity.
- Strict BitNet CUDA proof receipts name the official GGUF, explicit tokenizer,
  W1.58A8 layout, selected RTX 5070 Ti CUDA backend, 9950X3D CPU reference,
  fallback status, QK256 CUDA invocation counts, upload-once weight residency,
  and measured timing.
- Repeated strict ask benchmark receipts preserve same-model CPU AVX-512 and
  RTX 5070 Ti CUDA evidence before any speedup claim is accepted.
- CUDA-DENSE-001 remains an optional dense regular-LLM reference lane and is not
  part of completed BitNet packed-kernel proof.

## Hard Constraints

- CUDA visibility is not kernel execution.
- WGPU smoke is not CUDA proof.
- CPU fallback cannot count as CUDA execution.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.
- `speedup_claim` must remain false unless a same-model fallback-free benchmark
  receipt explicitly upgrades it.
- Dense regular-LLM CUDA receipts cannot satisfy BitNet packed I2S or QK256
  proof acceptance.

## Proof Ledger

RTX 5070 Ti CUDA BitNet proof state:

- strict selected backend: `nvidia-rtx-5070-ti-cuda`
- runtime API: `cuda`
- model: `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
- tokenizer: explicit external Llama BPE with BitNet.cpp answer template
- quantization/layout: `W1.58A8`, `gguf_packed_i2_s`
- CUDA kernel family: `qk256`
- weights uploaded once: `true`
- per-token weight upload: `false`
- QK256 CUDA kernel: `qk256_gemv_cuda`
- CUDA kernel invocations: `210` for one-token proof, `1680` for short decode
- BitNet linear CPU fallback: `0`
- fallback used: `false`
- speedup claim: `false`

Committed proof receipts:

- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-parity.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-proof.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-short-decode.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/strict-cuda-ask-math.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json`

Allowed claims:

- The RTX 5070 Ti CUDA selected backend executes receipt-backed CUDA kernels.
- Strict BitNet CUDA one-token and short-decode proofs run through QK256 CUDA
  kernels with zero BitNet linear CPU fallback.
- The current benchmark is a baseline with `speedup_claim=false`.

Not allowed:

- Do not claim dense regular-LLM CUDA proves BitNet packed inference.
- Do not claim WGPU, Vulkan, D3D12, or generic `cuda` as RTX 5070 Ti CUDA proof.
- Do not claim speedup unless a later same-model fallback-free receipt upgrades
  `speedup_claim` under the benchmark policy.

## Productization Follow-Up

The completed proof lane is not the same as a polished user-answer product path.
The next productization track is defined in
`docs/specs/rtx5070ti-cuda-answer-readiness.md`. `MODEL-ARTIFACT-007`,
`CPU-ANSWER-007`, and `CUDA-ANSWER-010` now provide the artifact, CPU, and strict
RTX 5070 Ti CUDA answer-corpus gates needed for the next step: CPU/CUDA answer
parity against the same artifact, tokenizer authority, prompt template, and
deterministic corpus.

Coherent CUDA answer claims still remain scoped. The current proof covers the
committed deterministic answer corpus, not broad chat quality, production server
readiness, or speedup.

`CUDA-ANSWER-011` records the first same-box CPU AVX-512 versus RTX 5070 Ti CUDA
answer-corpus comparison. Both backends pass the corpus, but the original
receipt preserved top-k logit divergence for every case and a `yes_no_water`
generated-answer divergence.

`CUDA-ANSWER-012` closes the generated-answer divergence by aligning the CUDA
QK256 inline-scale path with BitNet.cpp I2_S x I8_S activation semantics. CPU
AVX-512 and RTX 5070 Ti CUDA now match generated token IDs and decoded text for
all five committed deterministic corpus cases. Exact top-k logit parity remains
open for four cases.

`CUDA-PROD-001` started the user-path work after the strict corpus and parity
proofs. Strict `bitnet ask` now validates backend, fallback, and answer quality
even when the user omits `--receipt-out`; the default strict ask receipt still
preserves `speedup_claim=false`.

`CUDA-PROD-002` merged the active product-path gate: a strict CUDA warm session
that loads the model once, initializes CUDA once, uploads BitNet weights once,
serves multiple deterministic turns, and emits per-turn plus session-summary
receipts without broad chat, speed, server, or full-residency claims.

`CUDA-PROD-003` added explicit CUDA execution-residency coverage for the strict
answer path. `CUDA-PROD-004` added the strict answer-path benchmark baseline,
and `CUDA-PERF-001` added measured QK256 kernel time plus activation/output
transfer byte accounting. `CUDA-BITNET-PERF-002` starts the performance
qualification lane by recording repeated strict ask CPU AVX-512 and CUDA runs.
`CUDA-BITNET-PERF-003` extends that lane to repeated strict CUDA warm sessions
with model/tokenizer/context loaded once and upload-once QK256 handles. These
remain scoped measurements. `CUDA-BITNET-PERF-004` reviews that evidence and
keeps `speedup_claim=false` until decode-profile repetitions, transfer timing,
power/thermal context, and profile-specific acceptance thresholds are complete.
`CUDA-BITNET-PERF-005` is the next executable official BitNet performance
slice: repeated current-source profiles for one-token, short decode,
prefill/decode, warm-session, and warm-context decode. It must preserve
`speedup_claim=false`, `full_residency_claim=false`, `server_ready=false`, and
`dense_regular_llm_cuda_proof=false` while it gathers the missing profile
evidence.
`CUDA-PLANNER-001` adds the next contract layer: model-aware dispatch planning
that keeps BitNet QK256 CUDA and dense regular-LLM CUDA routes separate and
records unsupported strict CUDA routes instead of silently selecting CPU
fallback.
`CUDA-PLANNER-002` adds conservative model-family and quantization label mapping
into that planner contract, so recognized BitNet I2_S/QK256 and dense
regular-LLM FP16/BF16 metadata can route while unknown or mismatched metadata
stays unsupported under strict CUDA.
`CUDA-PLANNER-003` is the next receipt boundary: it summarizes planner decisions
into route-specific op counts, CPU fallback counts, unsupported counts, selected
route labels, and strict CUDA readiness without emitting those summaries from
real ask/session/benchmark receipts yet.
`CUDA-UX-001` starts the operator-facing proof cockpit: `bitnet receipts explain`
summarizes existing BitNet CUDA and dense regular-LLM CUDA receipts without
changing inference behavior or accepting new claims.
`CUDA-UX-002` reuses that receipt explanation layer for strict `bitnet ask`
proof summaries so the live user path prints the same route, backend, kernel,
fallback, timing, residency, and claim-limit fields as `receipts explain`.

Answer receipts must keep the completed proof invariants intact:

- selected backend: `nvidia-rtx-5070-ti-cuda`
- runtime API: `cuda`
- `qk256_gemv_cuda` invocation count greater than zero
- weights uploaded once: `true`
- per-token weight upload: `false`
- BitNet linear CPU fallback: `0`
- prompt prefill exercised: `true`
- answer quality gate passed: `true`
- speedup claim: `false` unless later benchmark-qualified

## Productization Current-State Ledger

The CUDA productization lane starts from existing receipts. This ledger records
the claim boundary before new runtime work.

| Lane | Current state | Last real receipt | Next missing proof |
|---|---|---|---|
| BitNet official 2B I2_S CUDA | product CLI ready, speed false | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json` | `CUDA-BITNET-PERF-005`: current-source repeated one-token, short-decode, prefill/decode, warm-session, and warm-context decode profiles before any speed, residency, or broad server promotion |
| Dense Qwen2.5 0.5B Q8_0 CUDA | product CLI ready in model coverage; strict one-token, short-decode, warm-session, benchmark-review, and exact-profile server-readiness receipts exist; persistent-handle receipt aliases now expose model/context/weight reuse; speed, full residency, broad dense GGUF, and broad server readiness remain false | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json` | logits/top-k transfer reduction, then governed speedup/full-residency review only with later exact-profile receipts |
| Qwen3 0.6B | product CLI ready for bounded ask/chat user paths after artifact, CPU sanity, all-layer plan, one-token, short-decode, warm-session, benchmark-review, ask user-path, chat user-path, and product-promotion review evidence; current-source non-streaming server-smoke evidence exists, but server_ready remains false until a separate exact-profile review accepts it; speed, server, full-residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet QK256 claims remain false | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json` | exact-profile server-readiness review; exact-profile performance comparator evidence before speed, benchmark-qualified, or full-residency promotion |
| SmolLM2 360M | structurally valid artifact contract; strict CPU retry reached one-token generation with `fallback_used=false` but failed the math quality gate; wrong-first-token diagnosis and comparator contract are recorded | `ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-reference-comparator-contract.json` | same-prompt first-token/top-k or checkpoint comparator capture before CPU answer readiness, all-layer planning, or CUDA |
| Llama 3.2 1B | registered candidate | none | artifact contract, tokenizer/prompt authority, CPU sanity |
| Llama 3.2 3B | registered candidate | none | memory envelope, artifact contract, tokenizer/prompt authority |
| Gemma/Phi small | registered candidate | none | architecture policy, artifact contract, tokenizer/prompt authority |

Allowed claims remain scoped to the receipts listed in this campaign and the
model coverage matrix. Dense Qwen proof is not BitNet proof. BitNet QK256 proof
is not dense SLM proof. Generic `cuda` is not RTX 5070 Ti proof unless the
receipt resolves it to `nvidia-rtx-5070-ti-cuda`.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| RTX5070TI-003 | merged | Preserved selected-device CUDA identity in #3679. |
| RTX5070TI-004 | merged | Added CUDA and NVML runtime probe in #3691. |
| RTX5070TI-005 | merged | Tiny CUDA kernel smoke receipt merged in #3723. |
| RTX5070TI-006 | merged | CPU/CUDA tiny fixture parity receipt merged in #3749. |
| RTX5070TI-007 | merged | CUDA receipt validation and kernel counters merged in #3756. |
| RTX5070TI-008 | merged | CUDA benchmark baseline merged in #3770. |
| CUDA-BITNET-001 | merged | Persistent CUDA BitNet context and weight handles merged in #3776. |
| CUDA-BITNET-002 | merged | Reusable CUDA I2S primitive merged in #3782. |
| CUDA-BITNET-003 | merged | QK256 fused dequant GEMV CUDA path merged in #3786. |
| CUDA-BITNET-004 | merged | CUDA BitNet upload-once weight handling merged in #3790. |
| CUDA-BITNET-005 | merged | BitNetLinear CUDA routing merged in #3792. |
| CUDA-BITNET-006 | merged | Strict one-token BitNet CUDA proof merged in #3801. |
| CUDA-BITNET-007 | merged | Strict short-decode BitNet CUDA proof merged in #3806. |
| CUDA-BITNET-008 | merged | Strict BitNet CUDA benchmark baseline merged in #3823. |
| CUDA-BITNET-009 | merged | Routed upload-once strict proof receipts merged in #3837. |
| CUDA-ANSWER-010 | merged | Strict RTX 5070 Ti CUDA answer corpus passes after the QK256 I2_S layout alignment in #4024. |
| CUDA-ANSWER-011 | merged | Same-box CPU AVX-512 and RTX 5070 Ti CUDA both pass the corpus, but the original receipt preserved top-k logit divergence and one generated-answer divergence. |
| CUDA-ANSWER-012 | merged | CUDA QK256 I8_S activation semantics close generated-token parity for all five committed corpus cases; exact top-k parity remains open. |
| CUDA-PROD-001 | merged | Strict `bitnet ask` validates through a default answer receipt when `--receipt-out` is omitted. |
| CUDA-PROD-002 | merged | Strict RTX 5070 Ti CUDA warm-session receipts with load/context/upload reuse across multiple deterministic turns. |
| CUDA-PROD-003 | merged | CUDA execution-residency coverage for the strict answer path. |
| CUDA-PROD-004 | merged | Strict answer-path benchmark baseline with `speedup_claim=false`. |
| CUDA-BITNET-PERF-002 | merged | Repeated strict ask benchmark receipts with same-model CPU AVX-512 and RTX 5070 Ti CUDA runs; `speedup_claim=false`. |
| CUDA-BITNET-PERF-003 | merged | Repeated strict CUDA warm-session benchmark receipts with load/context/upload reuse and measured QK256 timing/transfer counters; `speedup_claim=false`. |
| CUDA-BITNET-PERF-004 | merged | Benchmark qualification review for repeated strict ask and warm-session evidence; no profile upgraded, `speedup_claim=false`. |
| CUDA-BITNET-PERF-005 | ready | Next official BitNet performance slice: collect current-source repeated profiles for one-token, short-decode, prefill/decode, warm-session, and warm-context decode with strict `bitnet_qk256_cuda` routing, fallback false, QK256 kernel counters, timing/transfer/VRAM/power context, and all speed/full-residency/server/dense-proof claims still false. |
| CUDA-DENSE-001 | merged | Dense regular-LLM CUDA receipt boundary; not part of BitNet packed proof completion. |
| CUDA-DENSE-002 | merged | First dense CUDA FP16 GEMM smoke/parity fixture after the dense receipt boundary. |
| CUDA-DENSE-003 | merged | Dense regular-LLM CUDA tensor-residency receipt for the FP16 GEMM fixture; still no BitNet packed, dense GGUF inference, speedup, persistent session, or full-residency claim. |
| CUDA-DENSE-004 | merged | Persistent dense regular-LLM CUDA FP16 GEMM fixture session with one context/module, upload-once input buffers, repeated launches, and no dense GGUF inference, speedup, or full-residency claim. |
| CUDA-DENSE-005 | merged | Dense regular-LLM CUDA fixture receipts carry a model-aware `dense_regular_llm_cuda` execution plan while remaining rejected as BitNet packed I2_S/QK256 proof. |
| CUDA-PLANNER-001 | merged | Model-aware planner contract separating BitNet QK256 CUDA from dense regular-LLM CUDA and making unsupported strict CUDA fallback explicit. |
| CUDA-PLANNER-002 | merged | Conservative model-family and quantization metadata mapping into the model-aware planner spec. |
| CUDA-PLANNER-003 | merged | Receipt-ready planner summary fields for BitNet QK256 CUDA, dense regular-LLM CUDA, CPU fallback, unsupported ops, selected route labeling, and strict CUDA readiness. |
| CUDA-PLANNER-004 | merged | Real strict ask, answer-corpus, warm-session, and benchmark receipts emit model-aware `execution_plan` fields. |
| CUDA-UX-001 | merged | User-facing `bitnet receipts explain` command summarizes existing receipt proof fields, planner routes, kernels, timing, residency, and claim limits. |
| CUDA-UX-002 | merged | Strict `bitnet ask` proof summaries reuse the receipt explainer compact summary instead of a one-line bespoke formatter. |
| CUDA-DENSE-028 | merged | Promote verified dense GGUF `attention_v_mix` to the one-layer planner route while leaving `mlp_activation` as the remaining unsupported strict CUDA gap. |
| CUDA-DENSE-029 | merged | Extract a CPU-reference dense GGUF `mlp_activation` fixture while leaving CUDA parity, dense GGUF inference, Qwen token/decode/chat, speedup, and full-residency claims false. |
| CUDA-DENSE-030 | merged | Prove strict RTX 5070 Ti CUDA parity for the dense GGUF `mlp_activation` fixture while leaving route promotion, dense GGUF inference, Qwen token/decode/chat, speedup, and full-residency claims false. |
| CUDA-DENSE-031 | merged | Promote verified dense GGUF `mlp_activation` to the one-layer planner route, yielding 14 dense CUDA-routable ops, zero unsupported strict CUDA ops, and `strict_cuda_ready=true` without claiming dense inference, Qwen token/decode/chat, speedup, or full residency. |
| CUDA-DENSE-032 | merged | Define the governed full one-layer CPU reference harness contract for the verified Qwen2.5 0.5B Q8_0 dense GGUF artifact. |
| CUDA-DENSE-033 | merged | Implement the full layer-0 CPU reference harness so the next dense CUDA proof can compare the CUDA-routable one-layer plan against deterministic CPU-reference phase hashes and final output hash without claiming CUDA execution, dense inference, token/decode/chat, speedup, or full residency. |
| CUDA-DENSE-034 | merged | Define the integrated layer-0 CUDA parity contract so the next implementation can run the full CUDA-routable layer plan against the CUDA-DENSE-033 CPU reference without claiming dense inference, Qwen token/decode/chat, speedup, persistent/full residency, or BitNet packed proof. |
| CUDA-DENSE-035 | merged | Implement the integrated dense GGUF one-layer CUDA parity harness and receipt validator for the CUDA-DENSE-034 contract without claiming dense inference, Qwen token/decode/chat, speedup, persistent/full residency, or BitNet packed proof. |
| CUDA-DENSE-036 | merged | Define the governed dense GGUF all-layer execution-plan receipt contract so the next implementation must inspect every Qwen-family transformer layer, report route counts and graph differences, and keep model-boundary gaps explicit before any Qwen token/decode/chat claim. |
| CUDA-DENSE-PERF-001 | merged | Dense Qwen CUDA benchmark baseline receipt for one-token, short-decode, and warm-session profiles; `speedup_claim=false`. |
| CUDA-DENSE-PERF-002 | merged | Repeated same-artifact dense Qwen CPU/CUDA comparator receipts for the baseline profiles; still no speedup qualification. |
| CUDA-DENSE-PERF-003 | merged | Device-to-host logits download timing is measured in dense Qwen one-token, short-decode, and warm-session strict CUDA receipts; host-to-device timing remains explicitly unmeasured. |
| CUDA-DENSE-PERF-004 | merged | Dense Qwen benchmark qualification review consumes the baseline, repeated comparator, and D2H timing receipts while keeping speedup claims false. |
| CUDA-DENSE-PERF-005 | merged | Dense Qwen strict CUDA runtime receipts record a measured H2D model-load envelope, explicitly not pure CUDA event copy timing; `speedup_claim=false`. |
| CUDA-DENSE-PERF-006 | merged | Dense Qwen benchmark qualification consumes the H2D model-load envelope while preserving pure-H2D and speedup blockers. |
| CUDA-UX-006 | merged | `bitnet bench --device cuda --cuda-benchmark-receipt` reports governed CUDA benchmark receipts without simulating CPU fallback or claiming fresh benchmark execution. |
| CUDA-UX-007 | merged | `bitnet receipts explain` surfaces benchmark qualification profile decisions, transfer timing sources, and blockers for governed CUDA benchmark receipts. |
| CUDA-PROD-008 | merged | Reconcile the 5070 Ti BitNet and dense proof state before runtime changes. |
| CUDA-DENSE-050 | merged | Audit Qwen2.5 Q8_0 dense CUDA receipts to distinguish hardware/user-path evidence from validators and contracts. |
| CUDA-PROD-009 | merged | Harden strict BitNet CUDA ask/chat preflight with `bitnet cuda doctor`, fail-closed strict backend/tokenizer checks, visible receipt paths, and `speedup_claim=false`. |
| CUDA-PROD-010 | merged | Add a governed BitNet I2_S/QK256 CUDA product benchmark qualification receipt/report for five target profiles; report existing evidence through bench/receipts UX while keeping every speedup claim false. |
| CUDA-DENSE-051 | merged | Refresh or add the dense Qwen2.5 Q8_0 one-token strict CUDA hardware proof with explicit dense, non-BitNet claim boundaries; merged in #4645. |
| CUDA-DENSE-052 | merged | PR #4695 refreshed the current-source short-decode proof, superseded the stale-binary diagnostic blocker, and records fallback-free RTX 5070 Ti CUDA token parity with bounded decoded text `The answer is 4. What is`. |
| CUDA-DENSE-053 | merged | PR #4713 recorded the current-source dense Qwen2.5 Q8_0 warm-session strict CUDA proof with model/tokenizer/context loaded once, runtime buffer reuse, upload-once weights, generated-token equality, fallback false, and speed/full-residency/BitNet proof claims false. |
| CUDA-DENSE-054 | merged | PR #4720 recorded the current-source dense Qwen2.5 Q8_0 benchmark qualification review, consumed the one-token, short-decode, and warm-session receipts, rejected speedup for every reviewed profile, and kept BitNet QK256 proof false. |
| CUDA-DENSE-QWEN25-OPS-001 | merged | PR #5985 ranked Qwen2.5 dense CUDA bottlenecks from committed one-token, short-decode, warm-session, benchmark-review, H2D/D2H, and server readiness receipts; next work is persistent handles before logits/top-k transfer reduction. |
| CUDA-DENSE-QWEN25-OPS-002 | merged | PR #5995 exposed and validated persistent-handle receipt aliases for dense Qwen warm sessions: one model load, one CUDA context, upload-once weights, no per-request model load, workspace reuse, and fallback false. Speed, full residency, broad server readiness, and BitNet proof remain false; next work is logits/top-k transfer reduction. |
| CUDA-DENSE-QWEN25-OPS-003 | merged | PR #6010 exposed and validated Qwen2.5 logits-transfer accounting for short-decode and warm-session receipts. It records that D2H bytes remain full-logits because the CPU sampler still needs full logits until a device top-k sampler exists, while preserving selected-token equality, top-k evidence, quality evidence, and false speed/full-residency/BitNet claims. Next work is exact-profile requalification review. |
| CUDA-DENSE-QWEN25-PERF-007 | merged | Records the post-OPS-002/OPS-003 exact-profile Qwen2.5 requalification review. The review rejects speed and full-residency promotion because reviewed CUDA profiles remain slower than same-artifact CPU means, pure H2D timing is still unavailable, and logits-transfer accounting remains full-logits D2H until device top-k sampling exists. Exact-profile server readiness stays true; speedup, benchmark-qualified, full residency, broad dense GGUF, and BitNet proof remain false. |
| CUDA-MODEL-001 | merged | PR #4836 added the Qwen3 0.6B artifact contract as the first generalized dense model onboarding item without promoting CPU, CUDA, speed, server, or BitNet claims. |
| CUDA-MODEL-002 | merged | PR #4866 added the Qwen3 0.6B same-box CPU answer sanity receipt with the exact artifact contract, keeping CUDA, speed, server, product, full-residency, and BitNet QK256 claims false. |
| CUDA-MODEL-003 | merged | PR #4903 added the Qwen3 0.6B CUDA all-layer route plan after CPU sanity landed, keeping one-token CUDA, speed, server, full-residency, and BitNet QK256 claims false. |
| CUDA-MODEL-004 | merged | PR #4940 added the Qwen3 0.6B one-token strict CUDA proof after the all-layer plan landed, keeping short-decode, warm-session, speedup, server, full-residency, and BitNet QK256 claims false. |
| CUDA-MODEL-005 | merged | PR #4971 added the Qwen3 0.6B bounded short-decode strict CUDA proof after one-token CUDA landed, keeping warm-session, speedup, server, full-residency, broad dense GGUF, and BitNet QK256 claims false. |
| CUDA-MODEL-006 | merged | PR #4985 recorded Qwen3 0.6B warm-session strict CUDA proof after short-decode CUDA landed, keeping ask/chat product UX, speedup, server, full-residency, broad dense GGUF, and BitNet QK256 claims false. |
| CUDA-MODEL-007 | merged | PR #4996 recorded a Qwen3 0.6B benchmark qualification review for one-token, short-decode, and warm-session strict CUDA proof receipts, rejected speedup for every reviewed profile, and kept server, full-residency, broad dense GGUF, and BitNet QK256 claims false. |
| CUDA-MODEL-008 | merged | PR #5007 synced Qwen3 0.6B model coverage and model-status output to the exact earned accelerator-ready candidate tier after CUDA-MODEL-007, without promoting product CLI, speed, server, full-residency, broad dense GGUF, or BitNet QK256 claims. |
| CUDA-MODEL-009 | merged | PR #5507 audited Qwen3 0.6B product UX readiness and identified user-path ask/chat receipts plus product review as the remaining promotion blockers. |
| CUDA-MODEL-010 | merged | PR #5529 recorded the Qwen3 0.6B normal `bitnet ask` user-path receipt on RTX 5070 Ti CUDA with `fallback_used=false`, route `dense_regular_llm_cuda`, and no product, server, speedup, full-residency, or BitNet QK256 promotion. |
| CUDA-MODEL-011 | merged | PR #5645 recorded the Qwen3 0.6B normal `bitnet chat` user-path receipt with model/tokenizer/CUDA context/weights loaded once across multiple prompts while keeping product, server, speedup, full-residency, broad dense GGUF, and BitNet QK256 claims false. |
| CUDA-MODEL-012 | merged | PR #5660 reviewed and accepted Qwen3 0.6B product CLI promotion after the normal ask and chat user-path receipts. The accepted promotion is limited to bounded ask/chat CLI readiness and keeps server, speedup, benchmark-qualified, full-residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet QK256 claims false. |
| CUDA-MODEL-013 | merged | PR #5896 recorded the Qwen3 0.6B current-source shared-engine non-streaming `/v1/chat/completions` server-smoke receipt for the RTX 5070 Ti `dense_regular_llm_cuda` route while keeping server-ready, speedup, benchmark-qualified, full-residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet QK256 claims false. |
| CUDA-MODEL-014 | merged | PR #5866 reviewed Qwen3 exact-profile server readiness and rejected promotion at that point because a committed current-source non-streaming server-smoke receipt was still missing. The rejected review preserved every false speed, residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet proof claim. |
| CUDA-MODEL-014B | merged | PR #5918 accepted Qwen3 exact-profile server readiness only for `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`, preserving false speedup, benchmark-qualified speed, full residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet QK256 claims. |
| CUDA-MODEL-015 | merged | PR #5931 queued the repeated same-artifact Qwen3 CPU/CUDA comparator baseline for `one_token`, `short_decode_8`, `short_decode_32`, `warm_session_3_turns`, and `decode_128_from_warm_context`, noting that the existing single-run benchmark review is insufficient for speed or benchmark qualification. |
| CUDA-MODEL-016 | merged | PR #5941 added the Qwen3 repeated comparator receipt validator and generator for the five CUDA-MODEL-015 profiles. The contract requires at least three same-artifact CPU/CUDA runs per profile and preserves false speedup, benchmark-qualified speed, full residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet QK256 claims. |
| CUDA-MODEL-SMOLLM2-001 | merged | PR #5029 added the exact SmolLM2 360M Instruct Q8_0 artifact contract and report, promoted the coverage row only to structurally valid artifact-contract state, and kept CPU answer, CUDA, product CLI, speed, server, full-residency, broad dense GGUF, proof inheritance, and BitNet QK256 claims false. |
| CUDA-MODEL-SMOLLM2-002 | merged | PR #5048 synced the CUDA/model coverage view to the SLM-CPU-017 strict CPU preflight blocker: SmolLM2 360M reached strict CPU model-load preflight on the 9950X3D, then failed closed at the LayerNorm gamma guard before tokenizer/prompt/generation. The next proof is an SLM CPU loader policy follow-up, not CUDA. |
| CUDA-UX-008 | merged | PR #4724 added a CUDA model support dashboard sourced from the model coverage matrix. |
| CUDA-UX-009 | merged | PR #4754 added the strict RTX 5070 Ti BitNet CUDA user guide without changing product claims. |
| CUDA-UX-010 | merged | PR #4768 added the 9950X3D + RTX 5070 Ti CUDA quickstart after status and core proof surfaces were current. |
| CUDA-SERVER-001 | merged | PR #4820 added claim-safe strict dense Qwen server receipt classification without promoting server readiness or speed. |
| CUDA-SERVER-002 | merged | PR #4854 committed the exact bounded dense Qwen strict RTX 5070 Ti server-smoke receipt before any server-ready coverage promotion. |
| CUDA-SERVER-003 | merged | PR #5190 audited the bounded dense Qwen server-smoke receipt against BITNET-SPEC-0010 and recorded that `server_ready` remains false until a refreshed or supplemental receipt carries artifact checksum identity, endpoint/profile scope, and generation-policy fields. |
| CUDA-SERVER-004 | merged | PR #5393 hardened shared-engine server receipt fields and validation for exact-profile server-readiness evidence while keeping coverage promotion out of that slice. |
| CUDA-SERVER-005 | merged | PR #5410 promoted dense Qwen2.5 Q8_0 `server_ready=true` only for the exact non-streaming RTX 5070 Ti shared-engine `/v1/chat/completions` receipt, without speed, full-residency, deployment, concurrency, or BitNet proof claims. |
| CUDA-SERVER-006 | merged | PR #5439 added separate official BitNet I2_S/QK256 strict server-smoke evidence for the `bitnet_qk256_cuda` route while keeping broad production server readiness false. |
| CUDA-STATUS-001 | merged | PR #5158 added a user-facing CUDA capability matrix page backed by model coverage and campaign proof state without promoting model, speed, server, or residency claims. |

## Review Policy

CUDA PRs are non-stackable when they touch backend identity, kernels, receipts, or benchmark interpretation.
