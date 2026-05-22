# Kaby Lake SLM CPU Performance Dashboard

This dashboard is the baseline for i5-8250U dense SLM performance work. It
summarizes existing strict Qwen3-0.6B Q8_0 receipts only; it is not a sustained
throughput claim and it does not broaden support to Q4/Q5, server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256.

## Evidence Set

| Evidence | Path | Role |
| --- | --- | --- |
| Thread envelope | `ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-thread-timing-envelope.json` | 1, 2, 4, and 8 thread warm-session timing comparison |
| Thread validation | `ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-thread-timing-envelope-validation.json` | Validates strict provenance, quality, determinism, and no fallback across thread counts |
| Operator profile | `ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-operator-profile.json` | Default operator evidence with process memory, storage/free-space, warm-session timing, and unsupported-path fields |
| Greedy sampler fast path | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path-validation.json` | Validates that the guarded greedy no-penalty sampler fast path preserves the 4-thread generated IDs/text while sampler decode allocations remain zero |
| Logits extraction isolation | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse-validation.json` | Validates that direct tensor argmax bypasses full logits Vec extraction only where the sampler fast path is exact, while preserving generated IDs/text |
| Repetition-penalty logits reuse | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse-validation.json` | Validates that default repetition-penalty decode steps reuse a host logits scratch buffer instead of allocating fresh logits vectors, while preserving generated IDs/text |
| Warm-session sampler reuse | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-kv-temp-reuse-validation.json` | Validates that the temperature-zero warm-session profile reuses one sampler across prompts while preserving generated IDs/text and strict provenance |
| KV cache session reuse | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-kv-cache-session-reuse.json` | Records one CPU KV cache reused across prompts and cleared per prompt for prompt isolation |
| Prompt token cache | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json` | Validates that repeated rendered prompts reuse token IDs while preserving generated IDs/text and strict provenance |
| Prefill allocation attribution | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prefill-attribution-validation.json` | Validates that prompt prefill attribution preserves behavior and identifies `prompt_prefill.forward` as the dominant remaining allocation boundary |
| Packed Q8_0 sidecar runtime proof gate | `ci/slm-cpu/intel-i5-8250u/2026-05-19/qwen3-packed-q8-sidecar-runtime-proof-validation.json` | Records that packed Q8_0 sidecar runtime execution remains blocked because production dispatch still preserves eager F32 and no after-execution receipts exist |
| Post-bridge packed-Q8 counter classification | `ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-077-post-bridge-counter-classification.json` | Records real i5-8250U sidecar instrumentation counters and identifies `packed_matvec_compute` as the dominant exact-tensor sidecar cost |
| Post-aligned packed-Q8 matvec classification | `ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-079-post-aligned-matvec-classification.json` | Preserves the Qwen3 behavior oracle while recording a bounded counter-level `packed_matvec_ns` reduction against the SLM-CPU-077 sidecar oracle |

All rows use:

```text
model = Qwen/Qwen3-0.6B-GGUF / Qwen3-0.6B-Q8_0.gguf
sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
backend = cpu-rust
tokenizer.source = gguf_metadata
tokenizer.strict = true
fallback_used = false
prompt_template = qwen
qwen_no_think = true
temperature = 0.0
greedy = true
```

## Dashboard Refresh State

This refresh is current through SLM-CPU-079. It does not run new inference or
add a runtime optimization. It re-indexes the merged Kaby Lake Qwen3 Q8_0
evidence after KV-cache reuse, prompt-token caching, prefill attribution, and
the post-aligned exact-tensor packed-Q8 matvec artifact.

The current operator default remains evidence-scoped to the recorded 4-thread
operator profile. The default production runtime remains `eager_f32_candle`.
The packed-Q8 sidecar path remains opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight` / `blk.0.attn_q.weight`.

The current performance targets are:

1. Allocation/layout work around `prompt_prefill.forward` and `model.forward`,
   anchored by the SLM-CPU-035 prefill attribution receipt.
2. Exact-tensor packed-Q8 matvec compute work, anchored by the SLM-CPU-077
   counter pack and SLM-CPU-079 post-aligned counter classification.

Both targets remain gated by the Qwen3 Q8_0 behavior oracle: model SHA, strict
GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU
backend/kernel identity, dense hook identity where applicable, and
`fallback_used=false`.

## Thread Envelope

The thread envelope is a bounded single-run comparison. Generated IDs are stable
across the tested thread counts, and all tested runs preserve `fallback=false`.

| Threads | Total session ms | Warm prompt wall ms | Prefill ms | Decode total ms | First token mean ms | Steady decode mean tok/s | Warm generated tok/s | Cold generated tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 137179.478 | 94609.006 | 65837.089 | 23650.743 | 12311.833 | 1.931 | 0.486 | 0.335 |
| 2 | 136826.761 | 95562.894 | 65459.825 | 23672.087 | 12405.667 | 1.898 | 0.481 | 0.336 |
| 4 | 135679.052 | 94416.265 | 64733.698 | 23725.190 | 12328.333 | 1.963 | 0.487 | 0.339 |
| 8 | 136190.928 | 94979.978 | 65590.721 | 24318.391 | 12332.833 | 1.907 | 0.484 | 0.338 |

The best recorded total session and steady decode values in this bounded set are
the 4-thread run. The envelope still records `operator_default_recommendation =
null` because thermal, power, and resident memory were not sampled in the thread
envelope itself.

## Operator Profile Baseline

The operator profile fills the host-context gap for the selected operator shape.
It uses 4 threads and records process memory and storage/free-space context.

| Field | Value |
| --- | --- |
| Threads | 4 |
| Model loaded once | true |
| Tokenizer loaded once | true |
| Prompt count | 6 |
| Prompt tokens | 176 |
| Generated tokens | 46 |
| Total session ms | 141643.848 |
| Model load ms | 37531.292 |
| Tokenizer load ms | 780.975 |
| Warm prompt wall ms | 98140.650 |
| Prefill ms | 66958.577 |
| Decode total ms | 26077.955 |
| First token mean ms | 12709.333 |
| First token p95 ms | 13931.000 |
| Decode generated tok/s | 1.764 |
| Warm prompt generated tok/s | 0.469 |
| Cold session generated tok/s | 0.325 |
| Resident memory bytes | 3045429248 |
| Virtual memory bytes | 3036508160 |
| Model path free bytes | 45233147904 |
| Receipt path free bytes | 45233147904 |
| Thermal | unavailable, explicitly recorded |
| Power | unavailable, explicitly recorded |

The dashboard therefore treats 4 threads as the current operator-profile
default. That is not a sustained-performance recommendation; it is the only
default with both a thread-envelope comparison and an operator-profile receipt
containing memory/storage context.

## Current Hot-Loop Boundary

The operator profile already records several reuse decisions:

```text
model_loaded_once = true
tokenizer_loaded_once = true
session_owned_buffers = true
prompt_token_buffer_reused = true
generated_token_buffer_reused = true
stop_policy_precomputed_once = true
stop_tail_buffer_reused = true
timing_buffers_reused = true
allocation_audit_buffers_reused = true
kv_cache_recreated_per_prompt = false
kv_cache_reused_across_prompts = true
kv_cache_cleared_per_prompt = true
prompt_token_cache_enabled = true
prompt_token_cache_policy = rendered_prompt_token_ids_reused_across_repeated_warm_session_prompts
sampler_recreated_per_prompt = false
sampler_reused_across_prompts = true
logits_buffer_reuse_claimed = true
```

The dashboard originally identified KV cache recreation, repeated tokenization,
and logits scratch allocation as hot-loop risks. Later receipts narrowed those
without changing the Qwen3 Q8_0 behavior oracle:

| Slice | Evidence | Result |
| --- | --- | --- |
| `SLM-CPU-031` | `qwen3-kv-cache-session-reuse.json` | One session KV cache is reused across prompts and cleared per prompt; prompt isolation remains explicit. |
| `SLM-CPU-032` | `qwen3-prompt-token-cache-validation.json` | Repeated rendered prompt IDs are cached; prompt-tokenize allocation bytes drop from `1063162466` to `531586554`; generated outputs and strict provenance match the baseline. |
| `SLM-CPU-035` | `qwen3-prefill-attribution-validation.json` | `prompt_prefill_breakdown.embed` and `prompt_prefill_breakdown.forward` are present; first-prompt `prompt_prefill.forward` allocation is `554666438` bytes and dominates `prompt_prefill.embed` at `157108` bytes. |

The next safe optimization slices should therefore start from these known
remaining costs:

1. Reduce `prompt_prefill.forward` / `model.forward` allocation and layout churn
   using the typed transformer-forward workspace boundaries already introduced
   after SLM-CPU-035.
2. Continue reducing `model.logits` tensor allocation and output-head costs.
   SLM-CPU-026 removes fresh full logits Vec allocation from default
   repetition-penalty decode steps by reusing a host scratch buffer, but the
   model still produces logits tensors per token.
3. Continue keeping sampler, stop policy, and tokenizer work out of the hot loop
   where doing so preserves deterministic generated IDs. SLM-CPU-029 reuses one
   sampler across prompts for the temperature-zero Qwen3 profile only, and
   SLM-CPU-032 caches repeated rendered prompt token IDs for the bounded warm
   session. Nonzero temperature modes still recreate samplers to avoid
   RNG-state coupling.
4. Improve Q8_0 dense linear locality only with before/after receipts proving
   identical prompt IDs, generated IDs, decoded text, backend identity,
   tokenizer authority, model SHA, and `fallback=false`.

SLM-CPU-045 starts the production-facing sidecar carrier for that fourth
boundary. The GGUF loader now carries inert Q8_0 sidecar descriptors alongside
the existing eager F32 Candle tensors, recording packed block counts, payload
byte counts, tensor roles, source/runtime shapes, offsets, and packed-byte
hashes. The descriptors are metadata-only:

```text
eager_f32_runtime_preserved = true
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
generated_id_preservation_required_before_runtime_use = true
```

The next runtime hook must be a behavior-preserving dense-linear dispatch
selector. It cannot select packed Q8_0 sidecar compute until generated IDs,
decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel,
model SHA, and `fallback=false` match the established Qwen3 Q8_0 appliance
oracle.

SLM-CPU-046 starts that selector boundary. The selector contract makes the
current runtime choice explicit:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
sidecar_candidate_status = missing | present_but_unavailable
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
equivalence_gate_required = true
```

Packed Q8_0 sidecar descriptors may be visible to the selector, but they remain
unavailable for runtime compute until a later slice proves generated-ID/text and
strict-receipt equivalence. This is an API/receipt boundary, not a performance
claim.

SLM-CPU-047 adds the next gate between the fixture-level Q8_0 sidecar prototype
and the production selector. The gate can record that a packed sidecar fixture
matches the eager F32 fixture output within a bounded tolerance, but it still
keeps production sidecar runtime compute disabled until generated-ID/text and
strict-receipt equivalence exist:

```text
artifact_kind = dense_gguf_q8_sidecar_equivalence_gate
fixture_equivalence_passed = true | false
generated_id_receipt_equivalence_passed = false
sidecar_runtime_compute_allowed = false
selected_kernel = dense-f32-candle-linear
speedup_claim = false
```

This narrows the remaining runtime blocker without changing generated IDs,
decoded text, tokenizer authority, backend/kernel identity, model SHA, or
`fallback=false`.

SLM-CPU-048 keeps the next step non-executing as well: it consumes the
equivalence gate and emits a runtime preflight report that names the remaining
blockers before packed Q8_0 sidecar compute can be selected:

```text
artifact_kind = dense_gguf_q8_sidecar_runtime_preflight
selected_path = eager_f32_candle
fixture_equivalence_passed = true | false
generated_id_receipt_equivalence_passed = false
production_compute_hook_available = false
sidecar_runtime_compute_allowed = false
runtime_blockers = generated_id_receipt_equivalence_missing, production_compute_hook_missing, production_selector_still_eager_f32
```

The preflight is the final eligibility surface before a later production
compute hook can be attached. It still does not execute packed sidecar compute
or claim speedup.

SLM-CPU-049 adds the generated-ID/text receipt equivalence gate required before
that preflight can stop naming generated-ID evidence as a blocker. The gate
compares an eager F32 behavior oracle receipt against a future packed Q8_0
sidecar candidate receipt across the fields that matter for this lane:

```text
artifact_kind = dense_gguf_q8_generated_id_text_equivalence
model_sha256 = equal
tokenizer_source = equal
tokenizer_strict = true
corpus_id / prompt_id = equal
prompt_ids = equal
generated_ids = equal
decoded_text = equal
selected_backend = equal
selected_kernel = equal
fallback_used = false
speedup_claim = false
sidecar_runtime_compute_allowed = false
```

Passing this gate only proves behavior/provenance equivalence for the compared
receipts. It still keeps packed sidecar runtime compute disabled until a later
production compute hook exists and the selector is explicitly updated under the
same receipt discipline.

SLM-CPU-050 adds the production-compute-hook availability surface. This is
still a non-executing report: it distinguishes a missing production compute hook
from a hook that exists but is not yet selected by the runtime selector. The
selected path remains the eager F32 Candle oracle until a later selector update
is proven with generated-ID/text receipts:

```text
artifact_kind = dense_gguf_q8_production_compute_hook
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
hook_status = missing | available_but_selector_still_eager_f32
generated_id_receipt_equivalence_passed = true | false
production_compute_hook_available = true | false
selector_update_required_before_runtime_use = true
sidecar_runtime_compute_allowed = false
eager_f32_runtime_preserved = true
dense_runtime_replaced = false
speedup_claim = false
```

The availability surface does not execute packed Q8_0 sidecar compute, replace
the dense runtime, change generated IDs/text, or claim speedup. It only makes
the next runtime blocker machine-checkable.

SLM-CPU-051 adds the selector-readiness gate after the hook availability
surface. It is still non-executing: the report can state that generated-ID/text
equivalence and a production compute hook are both present, but it does not
change the runtime selector itself. A later selector-update PR must carry the
actual runtime behavior change and its before/after receipts.

```text
artifact_kind = dense_gguf_q8_selector_readiness_gate
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
readiness_status = blocked | ready_for_separate_selector_update
selector_update_ready = true | false
selector_update_required_before_runtime_use = true
sidecar_runtime_compute_allowed = false
runtime_blockers = production_selector_still_eager_f32 | ...
eager_f32_runtime_preserved = true
dense_runtime_replaced = false
speedup_claim = false
```

Readiness here means only that the next selector PR has the required proof
inputs. It is not packed Q8_0 production execution and it is not a speedup
claim.

SLM-CPU-052 is that next selector-update gate. It starts from the SLM-CPU-051
readiness artifact and adds an explicit selector-update receipt plus an opt-in
selector path for the packed Q8_0 sidecar candidate. The ordinary dense Q8_0
selector still preserves eager F32 Candle unless the selector-update proof is
supplied.

```text
artifact_kind = dense_gguf_q8_selector_update
previous_selected_path = eager_f32_candle
selected_path = packed_q8_sidecar | eager_f32_candle
selected_kernel = dense-q8-sidecar-linear | dense-f32-candle-linear
selector_update_applied = true | false
sidecar_runtime_compute_allowed = true | false
runtime_blockers = [] | ...
speedup_claim = false
```

The update may select the packed sidecar only where the eager F32 Candle oracle
and the packed Q8_0 candidate preserve the same prompt IDs, generated IDs,
decoded text, strict tokenizer authority, selected CPU backend/kernel, model
SHA, and `fallback=false`. It is a behavior-preserving selector update, not a
sustained-throughput or portable CPU-performance claim.

SLM-CPU-053 is the next runtime gate. It may enable packed Q8_0 sidecar compute
only for the exact evidence-scoped tensor or minimal tensor set covered by the
selector-update proof, and only with before/after receipts proving unchanged
model SHA, tokenizer source and strictness, prompt IDs, generated IDs, decoded
text, selected CPU backend/kernel identity, and `fallback=false`. If that proof
cannot be produced, the runtime must stay on eager F32 Candle and record the
specific blocker. This remains a behavior-preserving Kaby appliance slice, not
a sustained-throughput, Q4/Q5, accelerator, Qwen3.5, server, or BitNet QK256
claim.

The first SLM-CPU-053 runtime gate records the blocker instead of enabling
packed sidecar compute:

```text
artifact_kind = dense_gguf_q8_runtime_execution_proof
execution_status = blocked
selector_update_status = applied_to_packed_sidecar_candidate
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
production_runtime_hook_invoked = false
runtime_compute_enabled = false
sidecar_runtime_compute_allowed = false
runtime_blockers =
  production_dispatch_still_eager_f32
  packed_runtime_compute_disabled
  production_runtime_hook_missing
  before_after_receipts_missing
eager_f32_runtime_preserved = true
dense_runtime_replaced = false
fallback_used = false
speedup_claim = false
```

This closes only the runtime proof gate for the current state: packed Q8_0
sidecar compute is still disabled until a later PR adds the production forward
hook and after-execution receipts for the same Qwen3 Q8_0 behavior oracle.

SLM-CPU-054 makes that remaining hook/API gap machine-checkable instead of
treating it as an implicit prose blocker. It consumes the SLM-CPU-053 runtime
proof shape and records the exact production gaps that still prevent packed
Q8_0 sidecar compute from being selected:

```text
artifact_kind = dense_gguf_q8_runtime_hook_gap
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
production_runtime_hook_invoked = false
runtime_compute_enabled = false
sidecar_runtime_compute_allowed = false
api_gaps =
  production_dispatch_still_eager_f32
  packed_runtime_compute_still_disabled
  transformer_dense_linear_hook_missing
  before_after_receipt_capture_missing
fallback_used = false
speedup_claim = false
```

The next safe step remains a production dense-linear hook that receives the
selected Q8_0 sidecar descriptor and emits before/after Qwen3 Q8_0 behavior
receipts before enabling packed sidecar compute. SLM-CPU-054 does not replace
the eager F32 Candle runtime path and does not claim speedup.

SLM-CPU-055 is the next contract gate for that step. It should add the first
production dense-linear hook boundary so transformer dense linear calls can
receive either an explicit eager-F32 selection or a selected Q8_0 sidecar
descriptor. The default runtime must remain eager F32 Candle unless before/after
receipts prove identical model SHA, tokenizer authority, prompt IDs, generated
IDs, decoded text, selected CPU backend/kernel identity, and `fallback=false`.
If packed compute remains disabled, the slice must emit a machine-checkable
hook-contract or blocker artifact instead of making a speedup claim.

SLM-CPU-056 implements that hook boundary in the production transformer path.
Dense Q8_0 sidecar descriptors can now reach transformer attention and MLP
dense-linear calls as an explicit hook registry, and each call records whether
the selected descriptor is present. The runtime selection still remains:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
```

The machine-checkable artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-19/qwen3-production-dense-linear-hook-boundary.json`
records the remaining packed-compute and receipt gaps. The slice does not
enable packed Q8_0 sidecar execution or claim any performance improvement.

SLM-CPU-057 adds the next receipt gate on top of that boundary. Qwen3 Q8_0
warm-session aggregate and per-prompt receipts now carry a
`dense_q8_hook_selection` object so a future packed sidecar candidate must
preserve the hook-selection identity as well as model SHA, tokenizer
source/strictness, prompt IDs, generated IDs, decoded text, selected CPU
backend/kernel identity, and `fallback=false`.

The default runtime still remains:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
```

The machine-checkable blocker artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-19/qwen3-dense-hook-receipt-gate.json`
records that packed Q8_0 sidecar compute is still disabled until a packed
compute kernel exists and before/after Qwen3 Q8_0 warm-session receipts prove
identical generated output and dense hook-selection identity.

SLM-CPU-058 queued the before/after receipt gate. It records that packed Q8_0
sidecar compute remains disabled until a selected dense-hook path can be
compared against the eager F32 Candle behavior oracle by receipts that match on
model SHA, tokenizer source and strictness, prompt IDs, generated IDs, decoded
text, selected CPU backend/kernel identity, dense hook-selection identity, and
`fallback=false`. A speedup or sustained-throughput claim remains out of scope.

SLM-CPU-059 is the next compute-kernel proof gate. The committed blocker
artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-20/qwen3-packed-q8-compute-kernel-proof-gate.json`
names the current production gap: transformer dense-linear hooks still receive
metadata-only sidecar descriptors, not payload-bearing packed Q8_0 blocks, so
no production dense-linear call can safely execute a packed dequant-fused
matvec yet. The next safe runtime slice is to extend the hook contract with an
evidence-scoped payload-bearing sidecar candidate and prove one tensor path with
before/after Qwen3 Q8_0 warm-session receipts before broader selection or
timing claims.

SLM-CPU-060 adds that payload-bearing hook contract without enabling runtime
compute. `DenseLinearRuntimeHookDescriptor` can now carry an optional
`DenseLinearPackedQ8Payload` for a single evidence-scoped tensor path, and the
transformer boundary reports payload byte count, Q8 block count, matrix shape,
and contract validity. The selected production path remains `eager_f32_candle`
with `dense-f32-candle-linear`, `runtime_compute_enabled=false`,
`dense_runtime_replaced=false`, and `speedup_claim=false`. The committed
contract artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-20/qwen3-payload-bearing-q8-sidecar-hook-contract.json`
records the remaining gate: production loading still attaches metadata-only
descriptors by default, and one exact Qwen3 Q8_0 tensor path must be wired
behind an opt-in gate and proved by before/after warm-session receipts before
packed Q8_0 sidecar compute can be selected.

SLM-CPU-061 wires one exact real Qwen3 Q8_0 dense-linear tensor into that
payload-bearing hook contract behind an explicit opt-in gate. With
`BITNET_DENSE_Q8_PAYLOAD_ENABLE=1` and
`BITNET_DENSE_Q8_PAYLOAD_TENSOR=blk.0.attn_q.weight`, the production GGUF load
attaches the packed bytes for `layers.0.attention.q_proj.weight` to a single
`AttentionQ` hook boundary. The committed before/after warm-session receipts
show the selected path remains `eager_f32_candle`, selected kernel remains
`dense-f32-candle-linear`, `runtime_compute_enabled=false`,
`dense_runtime_replaced=false`, and generated outputs are unchanged. The
summary artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-20/qwen3-real-q8-sidecar-payload-candidate.json`
records matching model SHA, strict GGUF tokenizer authority, prompt IDs,
generated IDs, decoded text, selected CPU backend/kernel identity, and
`fallback=false`. It does not claim packed Q8 compute, speedup, sustained
throughput, Q4/Q5 support, accelerator execution, Qwen3.5 support, or BitNet
QK256 changes.

## Greedy Sampler Fast Path

SLM-CPU-024 adds a guarded sampler fast path for `temperature = 0.0` when
repetition penalty is inactive, or when there is no context to penalize. The
sampler returns the greedy argmax directly instead of copying logits into its
scratch buffer first.

The after-change validation compares the new 4-thread warm-session receipt
against the SLM-CPU-015 4-thread baseline:

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-warm-session-threads-4.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path-validation.json
generated_outputs_match_baseline = true
sampler_decode_allocations_zero = true
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This closes only the greedy no-penalty sampler scratch-copy boundary. It does
not remove the remaining `model.logits_and_extract` allocation, change Q8_0
dense math, or establish a sustained throughput claim.

## Logits Extraction Isolation

SLM-CPU-025 narrows the next allocation boundary without changing generated
tokens. In the guarded deterministic greedy/no-penalty case it selects the
argmax directly from the logits tensor, bypassing full host `Vec<f32>` logits
extraction. For default warm-session steps with active repetition penalty, it
keeps the existing vector extraction path because that path is still required
for count-aware repetition-penalty semantics.

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse-validation.json
generated_outputs_match_baseline = true
direct_greedy_logits_steps = 6
logits_vec_extraction_steps = 40
logits_vec_extraction_bypassed_for_all_steps = false
model_logits_and_extract_alloc_bytes_delta = -3643896
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This is an isolation slice, not a full logits-buffer reuse claim. The remaining
vector extraction steps are explicit in the receipt and should only be removed
after the repetition-penalty path has an allocation-safe equivalent that
preserves generated IDs.

## Repetition-Penalty Logits Reuse

SLM-CPU-026 adds that allocation-safe repetition-penalty equivalent for the
warm-session path. Count-aware repetition penalties are still applied before
greedy selection, but default decode steps now copy CPU F32 logits into a
caller-owned scratch buffer and sample in place instead of materializing a fresh
host `Vec<f32>` per token.

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse-validation.json
generated_outputs_match_baseline = true
direct_greedy_logits_steps = 6
logits_scratch_reuse_steps = 40
logits_vec_extraction_steps = 0
logits_vec_extraction_bypassed_for_all_steps = true
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This still does not claim sustained throughput or dense math acceleration. It
only narrows the host allocation boundary for the existing Qwen3 Q8_0 CPU
behavior oracle.

## Warm-Session Sampler Reuse

SLM-CPU-029 removes the remaining per-prompt sampler object recreation in the
bounded Qwen3 Q8_0 warm-session appliance profile. The reuse is deliberately
guarded to `temperature = 0.0`, where the sampler does not need cross-prompt RNG
state. Sampling modes with nonzero temperature retain per-prompt sampler
creation for request independence.

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-kv-temp-reuse.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-kv-temp-reuse-validation.json
generated_outputs_match_baseline = true
sampler_recreated_per_prompt = false
sampler_reused_across_prompts = true
sampler_reused_prompt_count = 6
sampler_recreated_prompt_count = 0
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This slice does not change KV-cache isolation: the KV cache is still recreated
per prompt to preserve prompt independence. It also does not claim a measured
speedup; allocation counters for the dominant tensor-producing components are
unchanged, and the evidence is scoped to removal of avoidable sampler setup in
the existing 4-thread appliance profile.

## Warm-Session KV Cache Reuse

SLM-CPU-031 reuses one CPU KV cache across Qwen3 Q8_0 warm-session prompts and
clears it before each prompt. The reuse is scoped to the resident session and
keeps prompt isolation explicit; it moves the large KV-cache tensor allocation
out of per-prompt `prompt_setup` and records it once as session setup.

```text
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-kv-cache-session-reuse.json
generated_outputs_match_baseline = true
quality_passed = true
determinism_passed = true
kv_cache_recreated_per_prompt = false
kv_cache_reused_across_prompts = true
kv_cache_cleared_per_prompt = true
kv_cache_reused_prompt_count = 6
session_setup_kv_cache_alloc_bytes = 9395257760
prompt_setup.kv_cache_alloc_bytes_per_first_prompt = 0
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This does not claim sustained throughput or a portable performance result. It
only narrows the resident-session allocation boundary for the existing Qwen3
Q8_0 4-thread Kaby appliance profile.

## Prompt Token Cache

SLM-CPU-032 reuses rendered prompt token IDs inside the resident warm session.
The reuse is scoped to repeated prompts with the same rendered prompt, BOS
policy, and special-token parse policy. It keeps the generated-ID oracle intact
while avoiding redundant `tokenizer.encode` work for repeated corpus cases.

```text
prompt_token_cache_policy = rendered_prompt_token_ids_reused_across_repeated_warm_session_prompts
prompt_token_cache_enabled = true
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json
prompt_token_cache_hits = 3
prompt_token_cache_misses = 3
prompt_token_cache_entry_count = 3
generated_outputs_match_baseline = true
quality_passed = true
determinism_passed = true
fallback_used = false
before_prompt_tokenize_alloc_bytes = 1063162466
after_prompt_tokenize_alloc_bytes = 531586554
speedup_claim = false
sustained_throughput_claim = false
```

This is a tokenizer-boundary optimization only. It does not change dense math,
does not claim Q4/Q5 support, and does not turn the bounded Kaby appliance
profile into a sustained-throughput result.

## Aggregate Allocation Boundary

SLM-CPU-033 records the next warm-session allocation target directly in the
aggregate receipt. After prompt-token caching, the real i5-8250U evidence still
shows the largest allocation-counter deltas in prompt prefill and dense model
forward execution:

```text
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json
dominant_hotspot = prompt_prefill
next_optimization_target = prefill_model_forward_allocation_attribution
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

The target is diagnostic. It tells the next runtime slice to attribute
`prompt_prefill` and `model.forward` tensor allocation before changing kernels
or dense math. It does not prove a speedup and it does not broaden the Kaby
claim boundary.

## Prompt Prefill Attribution

SLM-CPU-034 keeps the existing `prompt_prefill` allocation total for receipt
continuity and adds nested subcomponent counters:

```text
prompt_prefill_breakdown.embed
prompt_prefill_breakdown.forward
ranked_hotspots includes prompt_prefill.embed and prompt_prefill.forward
```

This is attribution only. It identifies whether the prefill hotspot is coming
from token embedding or dense model forward execution before any Q8_0 GEMV,
RMSNorm, RoPE, or attention-loop optimization is attempted.

## Prefill Attribution Artifact

SLM-CPU-035 records a real i5-8250U Qwen3 Q8_0 warm-session receipt after the
prompt-prefill attribution slice:

```text
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prefill-attribution.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prefill-attribution-validation.json
generated_outputs_match_baseline = true
decoded_text_matches_baseline = true
fallback_used = false
tokenizer_source = gguf_metadata
tokenizer_strict = true
first_prompt_prefill_embed_alloc_bytes = 157108
first_prompt_prefill_forward_alloc_bytes = 554666438
```

The receipt preserves the behavior oracle from the prompt-token cache baseline
while adding `prompt_prefill_breakdown.embed`,
`prompt_prefill_breakdown.forward`, and ranked prompt-prefill subcomponent
hotspots. It is evidence for attribution only: it does not claim a runtime
speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

## Prefill Forward Buffer Boundary

SLM-CPU-036 classifies the first reusable allocation surface under
`prompt_prefill.forward` before changing dense math. The warm-session receipt
first recorded the boundary as transformer forward workspace and owned tensor
outputs that were not caller-reusable from the CLI warm-session loop:

```text
next_optimization_target = prefill_forward_buffer_boundary
first_reusable_allocation_surface = transformer_forward_workspace_api_and_owned_tensor_outputs
claim_scope = allocation-boundary classification only
```

SLM-CPU-037 keeps that boundary explicit instead of adding a broad caller-side
buffer reuse patch. SLM-CPU-038 introduces the first typed transformer forward
workspace API boundary through `TransformerModel::forward_with_workspace`,
`TransformerBlock::forward_with_workspace`, and
`FeedForward::forward_with_workspace`. This first API slice still delegates
tensor math to the existing owned-output path, so reusable tensor storage is not
enabled yet. The receipt therefore records:

```text
required_api_boundary = typed_transformer_forward_workspace
optimization_deferred = true
```

SLM-CPU-039 narrows that boundary to the first workspace-owned transformer
output surface. `FeedForward::forward_with_workspace` now routes the owned
feed-forward output tensor through `TransformerForwardWorkspace` before
returning it to the existing block math. Candle still constructs the tensor
through the existing owned-output path, so this is ownership plumbing and
allocation attribution, not reusable storage or a speedup claim:

```text
first_reusable_allocation_surface = feed_forward.output
workspace_storage_owner = TransformerForwardWorkspace
reuse_status = feed_forward_output_workspace_owned_reuse_not_enabled
next_optimization_target = feed_forward_output_workspace_owned_boundary
status = workspace_owned_output_reuse_deferred
optimization_deferred = true
```

SLM-CPU-040 reaches the next narrower hook: the exact
`FeedForward::down_proj` output boundary. The workspace observes this boundary
before the output is returned, but reusable storage remains blocked because
`candle_nn::Linear::forward` constructs and returns a new `Tensor`; it does not
expose an out-parameter or reusable output-storage API that can be filled by
`TransformerForwardWorkspace` without changing the linear implementation.

```text
first_reusable_allocation_surface = feed_forward.down_proj.output
workspace_storage_owner = TransformerForwardWorkspace
reuse_status = feed_forward_down_proj_output_storage_reuse_blocked_by_candle_linear
next_optimization_target = feed_forward_down_proj_output_storage_boundary
status = down_proj_output_storage_reuse_blocked
optimization_deferred = true
```

The next safe implementation target is adding or adopting a
behavior-preserving linear output-storage API before replacing
`FeedForward::down_proj` output construction with reusable workspace-backed
storage. This remains an allocation-boundary proof only; it makes no speedup,
sustained-throughput, broad-answer-quality, Q4/Q5 runtime, accelerator,
Qwen3.5, or BitNet QK256 claim.

SLM-CPU-041 narrows the blocker from "Linear is opaque" to the exact Candle
tensor API gap. `candle_nn::Linear` exposes `weight()` and `bias()`, so the
Qwen3 `FeedForward::down_proj` shapes can be inspected without changing math.
The behavior-preserving compute path still uses `Tensor::matmul` plus optional
`broadcast_add`, and those operations return owned tensors without a
caller-provided output-storage parameter. Reusable workspace-backed storage is
therefore still deferred until Candle grows, or this repo adopts, an explicit
matmul/bias-add output-storage API:

```text
first_reusable_allocation_surface = feed_forward.down_proj.output
workspace_storage_owner = TransformerForwardWorkspace
reuse_status = dense_linear_output_storage_blocked_by_candle_tensor_ops
required_api_boundary = dense_linear_output_storage_api_boundary
weight_accessible = true
bias_accessible = true
can_fill_caller_output_storage = false
optimization_deferred = true
```

This is a classification slice. It does not change Q8_0 GEMV, RMSNorm, RoPE,
attention, output-head math, tokenizer behavior, or generated tokens, and it
does not claim a speedup, sustained throughput, Q4/Q5 runtime support,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

SLM-CPU-042 moves the next target from reusable output storage to the first
Q8_0 dense linear locality boundary that can be inspected without changing
runtime behavior. The current dense standard GGUF load path eagerly dequantizes
Q8_0 blocks into a host `Vec<f32>`, reshapes projection and embedding tensors
for Candle without transposing the dequantized values, and then constructs a
Candle tensor from that F32 slice:

```text
locality_boundary = eager_dense_standard_quant_dequant_to_f32_before_candle_tensor
dequantizes_before_compute = true
materializes_f32_tensor = true
values_transposed = false
shape_reshaped_without_transpose = true for GGML projection/embedding layouts
next_optimization_target = q8_dense_linear_locality_boundary
```

This boundary is instrumentation only. A later runtime optimization may replace
the eager F32 materialization or improve Q8_0 dense linear locality only if it
preserves generated IDs, decoded text, strict GGUF tokenizer authority, selected
CPU backend/kernel, model SHA, and `fallback=false` in before/after receipts.
It does not reopen the Candle output-storage blocker, claim a speedup, or
broaden support to Q4/Q5, accelerators, Qwen3.5, or BitNet QK256.

SLM-CPU-043 starts that later slice with a fixture-level Q8_0 sidecar
prototype. The prototype keeps packed Q8_0 block scales and signed codes,
computes a narrow CPU matvec by dequantizing inside the dot product, and
compares the result against the existing eager F32 fixture output:

```text
artifact_kind = dense_gguf_q8_linear_sidecar_prototype
dequantizes_inside_matvec = true
materializes_full_f32_weights = false
compares_against_eager_f32_reference = true
dense_runtime_replaced = false
speedup_claim = false
```

This is not yet a production transformer path. It proves the first local
packed-Q8 sidecar compute boundary can match the eager F32 fixture for a single
dense linear role. A runtime replacement still requires before/after warm-session
receipts proving unchanged generated IDs, decoded text, strict tokenizer
authority, selected CPU backend/kernel, model SHA, and `fallback=false`.

SLM-CPU-067 promotes that boundary to one exact real Qwen3 tensor path:
`layers.0.attention.q_proj.weight`, sourced from `blk.0.attn_q.weight`. The
runtime hook remains opt-in and exact-tensor gated. By default, the transformer
continues to use the eager F32 Candle linear path. Packed sidecar execution is
selected only when all four environment gates name the same evidence-scoped
tensor:

```text
BITNET_DENSE_Q8_PAYLOAD_ENABLE=1
BITNET_DENSE_Q8_PAYLOAD_TENSOR=blk.0.attn_q.weight
BITNET_DENSE_Q8_RUNTIME_ENABLE=1
BITNET_DENSE_Q8_RUNTIME_TENSOR=blk.0.attn_q.weight
```

The SLM-CPU-067 Kaby artifact bundle records before/after strict warm-session
receipts for the verified Qwen3 Q8_0 appliance profile:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-067-before-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-067-after-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-067-runtime-hook-equivalence.json
```

The baseline receipt reports `selected_path = eager_f32_candle`. The opt-in
receipt reports `selected_path = packed_q8_sidecar`,
`selected_kernel = dense-q8-sidecar-linear`, and
`runtime_compute_enabled = true` for only the exact Q projection tensor. The
equivalence artifact records matching prompt IDs, generated IDs, decoded text,
model SHA, strict GGUF tokenizer authority, selected CPU backend, and
`fallback=false` across both receipts.

This is a behavior-preserving exact-tensor runtime hook candidate. It does not
enable packed Q8_0 runtime compute by default, does not generalize packed Q8_0
execution to all dense tensors, and does not claim speedup, sustained
throughput, broad answer quality, Q4/Q5 runtime support, accelerator execution,
Qwen3.5 support, or BitNet QK256 changes.

SLM-CPU-068 classifies the timing evidence from that exact same artifact pack.
The result is intentionally conservative: the hook preserved generated IDs and
decoded text, but the opt-in packed sidecar path regressed on the bounded
two-prompt 4-thread receipt.

```text
classification = regressed_on_bounded_two_prompt_artifact
before_selected_path = eager_f32_candle
after_selected_path = packed_q8_sidecar
behavior_equivalence.passed = true
runtime_promotion_recommended = false
speedup_claim = false

before.total_session_ms = 118266.381
after.total_session_ms = 138900.052
delta.total_session_ms = +20633.671 (+17.446%)

before.warm_prompt_wall_ms = 41085.124
after.warm_prompt_wall_ms = 57917.182
delta.warm_prompt_wall_ms = +16832.058 (+40.966%)

before.decode_generated_tok_s = 1.225
after.decode_generated_tok_s = 0.956
delta.decode_generated_tok_s = -0.269 (-21.959%)
```

The timing classification artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-068-exact-hook-timing-classification.json`.
The default runtime should remain `eager_f32_candle`. The packed sidecar hook is
still useful as a correctness-preserving implementation boundary for further
locality and kernel work, but it is not a promotion candidate in its current
form.

SLM-CPU-069 localizes the next packed-Q8 sidecar work from that regression
without changing runtime defaults. The machine-checkable root-cause artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-069-packed-q8-locality-root-cause.json`.
It records that the regression is concentrated in prefill/forward timing while
logits and sampling are stable, and classifies the likely root cause as
`packed_block_decode_and_matvec_locality`.

The current opt-in sidecar kernel is a scalar reference matvec over the packed
Q8_0 bytes for only `layers.0.attention.q_proj.weight`. It decodes the fp16
Q8_0 block scale inside the innermost per-weight loop and does not reuse the
scale across the 32 codes in each block. The next safe target is therefore a
block-local matvec prototype that decodes each block scale once per block and
keeps behavior gated against the eager F32 oracle before any timing
interpretation.

```text
root_cause.primary = packed_block_decode_and_matvec_locality
root_cause.secondary = scratch_allocation
root_cause.unlikely = selector_overhead, receipt_timing_instrumentation
next_target = SLM-CPU-070 packed Q8 block-local matvec prototype
default_runtime_changed = false
speedup_claim = false
```

SLM-CPU-070 adds that prototype in the exact-tensor opt-in sidecar helper
without promoting it to the default runtime. The implementation now walks
contiguous Q8_0 block spans and decodes each fp16 scale once per block segment
instead of once per individual weight. The code-level artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-070-packed-q8-block-local-matvec-prototype.json`.

The prototype remains a behavior-gated implementation slice, not a performance
claim. The verified Qwen3 GGUF was not present in the development workspace for
this slice, so the committed evidence is limited to the exact dense-linear
reference tests. Real i5-8250U before/after receipt equivalence is still
required before any timing interpretation or promotion decision.

```text
selected_default_path = eager_f32_candle
opt_in_candidate_path = packed_q8_sidecar
exact_tensor = layers.0.attention.q_proj.weight
scale_decode = once per contiguous Q8 block segment
code_level_reference_tests = passed
row_split_q8_block_reference_test = passed
real_qwen3_generated_id_receipts_regenerated = false
speedup_claim = false
```

SLM-CPU-071 is the required real-artifact gate after that prototype. It must
regenerate the verified Qwen3-0.6B Q8_0 i5-8250U 4-thread warm-session
before/after receipts with the default eager F32 path and the opt-in exact
`layers.0.attention.q_proj.weight` packed sidecar candidate. The first gate is
behavioral: prompt IDs, generated IDs, decoded text, strict GGUF tokenizer
authority, selected CPU backend/kernel, model SHA, hook identity, and
`fallback=false` must match before timing is classified as improved, regressed,
or inconclusive. This gate still does not enable packed Q8_0 by default or claim
sustained throughput.

SLM-CPU-071 merged the gate definition, not the actual before/after artifact
pack. No `qwen3-slm-cpu-071-*` warm-session receipts are committed under
`ci/slm-cpu/intel-i5-8250u/2026-05-21/`. SLM-CPU-072 is therefore the real
artifact-capture follow-up: it must either commit the before/after receipts and
classification for the exact Qwen3 Q8_0 appliance profile, or record that the
verified GGUF was unavailable in the execution environment. It must not treat
the merged gate definition as timing evidence.

## Claim Boundary

This dashboard may be used to claim:

```text
Qwen3-0.6B Q8_0 has a bounded i5-8250U strict CPU performance baseline.
The current operator profile uses 4 threads.
Generated IDs were stable across the 1/2/4/8 thread envelope.
Memory and storage context are present for the 4-thread operator profile.
Thermal and power fields are present but unavailable.
```

This dashboard must not be used to claim:

```text
sustained 8250U throughput
broad chat quality
Q4/Q5 support
server inference
GPU, NPU, OpenVINO, or UHD 620 acceleration
Qwen3.5 support
BitNet QK256 changes
portable performance across other CPUs
```

## Release-Surface Boundary

After SLM-CPU-061, BitNet-rs keeps the Kaby Lake SLM lane as the audited
release and evidence surface. The committed Qwen3 Q8_0 and Qwen2.5 Q8_0
strict CPU receipts remain the behavior oracle for any future dense packed-Q8
candidate.

Further packed Q8_0 compute-candidate development should happen in
`bitnet-rs-swarm`. A candidate can return to BitNet-rs only as an audited
release/evidence artifact that preserves:

```text
model SHA
strict GGUF tokenizer authority
prompt IDs
generated IDs
decoded text
selected CPU backend/kernel identity
dense hook-selection identity
fallback_used=false
speedup_claim=false unless a separate bounded timing receipt proves otherwise
```

This boundary does not claim a new runtime compute path, speedup, sustained
throughput, broad answer quality, Q4/Q5 runtime support, server execution,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

The intake rules for those returned artifacts are defined in
`docs/slm/SLM_CPU_SWARM_ARTIFACT_INTAKE.md`.

## SLM-CPU-065 Runtime Promotion Gate

SLM-CPU-065 reviewed the accepted single-tensor packed Q8_0 sidecar candidate
from the SLM-CPU-064 intake package:

```text
tensor = layers.0.attention.q_proj.weight
role = AttentionQ
payload_sha256 = a8e16a232ea8c19a5c5d2eb5f21bfdf5c297eba0ac90e74b0afc052577179c24
```

The release-surface decision is blocked, not promoted. BitNet-rs keeps the
runtime on the strict eager F32 Candle path until an exact-tensor packed-Q8
runtime receipt exists:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
runtime_compute_enabled = false
dense_runtime_replaced = false
fallback_used = false
speedup_claim = false
```

The gate artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-20/slm-cpu-065-runtime-promotion-gate.json`.
It records that the sidecar evidence was accepted, but runtime promotion is
blocked because the release surface has not produced a strict before/after
receipt where packed sidecar compute is selected for this exact tensor and
behavior remains identical.

This gate does not claim packed Q8_0 execution, speedup, sustained throughput,
broad answer quality, Q4/Q5 runtime support, server execution, accelerator
execution, Qwen3.5 support, or BitNet QK256 changes.

## SLM-CPU-074 Packed Q8 Sidecar Instrumentation

SLM-CPU-074 adds aggregate runtime counters for the opt-in exact-tensor
`layers.0.attention.q_proj.weight` packed Q8_0 sidecar path. The counters are
available from `bitnet-transformer` through
`dense_q8_sidecar_instrumentation_snapshot()` and can be reset with
`reset_dense_q8_sidecar_instrumentation()`.

The instrumentation measures the costs that SLM-CPU-073 identified as the next
blocking surface:

```text
selector dispatch calls and elapsed ns
selector selected / declined / error counts
input materialization calls, elapsed ns, and value count
bias extraction calls, elapsed ns, and value count
packed matvec calls, elapsed ns, input rows, and output values
output tensor construction calls and elapsed ns
```

The default production path remains `eager_f32_candle`. Packed Q8_0 sidecar
execution remains opt-in, payload-gated, and exact-tensor scoped. These counters
are diagnostic evidence only; they do not claim speedup, sustained 8250U
throughput, broad answer quality, Q4/Q5 runtime support, server execution,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

## SLM-CPU-075 Instrumentation Artifact Boundary

SLM-CPU-075 records the first post-instrumentation diagnostic artifact:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-075-packed-q8-instrumentation-artifact.json
```

The artifact consumes the SLM-CPU-074 counter surface and keeps the prior
SLM-CPU-072 before/after receipts as the behavior oracle. It classifies selector
dispatch, input materialization, bias extraction, packed matvec compute, and
output tensor construction as instrumented surfaces, but it does not claim real
end-to-end counter values because the warm-session receipt path does not yet
snapshot and serialize the transformer-side aggregate counters around a bounded
Qwen3 Q8_0 run.

The resulting blocker is narrow: add a warm-session sidecar instrumentation
receipt bridge before using these counters to drive another packed-Q8 runtime
promotion or optimization. That bridge must reset the counters before the
opt-in exact-tensor run, snapshot them afterward, and prove generated IDs,
decoded text, model SHA, tokenizer source and strictness, selected CPU backend
identity, dense hook identity, and `fallback_used=false` remain unchanged.

The default path remains `eager_f32_candle`; packed Q8_0 sidecar execution
remains opt-in and exact-tensor scoped to `layers.0.attention.q_proj.weight`.
SLM-CPU-075 makes no speedup, sustained-throughput, broad answer-quality,
Q4/Q5 runtime-support, server, accelerator, Qwen3.5, or BitNet QK256 claim.

## SLM-CPU-076 Warm-Session Instrumentation Bridge

SLM-CPU-076 bridges the packed Q8_0 sidecar instrumentation counters into the
Qwen3 Q8_0 warm-session aggregate receipt. The warm-session command resets the
`bitnet-transformer` counters before the prompt loop and snapshots them after
the bounded run, then records the result under
`dense_q8_sidecar_instrumentation`.

The receipt bridge serializes:

```text
selector dispatch / selected / declined / error counters
input materialization calls, elapsed ns, and value count
bias materialization calls, elapsed ns, and value count
packed matvec calls, elapsed ns, input rows, and output values
output tensor construction calls and elapsed ns
```

It also records that the sidecar path remains opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight`, while the default runtime remains
`eager_f32_candle`. The bridge is diagnostic only. It does not enable packed
Q8_0 sidecar execution by default, broaden the hook beyond the exact tensor,
or claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime
support, server execution, accelerator execution, Qwen3.5 support, or BitNet
QK256 changes.

## SLM-CPU-077 Post-Bridge Counter Artifact

SLM-CPU-077 consumes that bridge with a real i5-8250U Qwen3 Q8_0 warm-session
sidecar run. The committed artifact pack is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-077-post-bridge-packed-q8-sidecar-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-077-post-bridge-counter-classification.json
```

The sidecar run remains opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight`. The companion classification compares the
new sidecar receipt against the committed SLM-CPU-072 eager F32 oracle because
a same-turn eager oracle rerun failed on the low-free-space Kaby host before
receipt write with:

```text
memory allocation of 167772160 bytes failed
```

The behavior oracle still passes for the compared receipts:

```text
prompt_ids_match = true
generated_ids_match = true
decoded_text_match = true
model_sha_match = true
tokenizer_source_match = true
tokenizer_strict_match = true
selected_backend_match = true
fallback_false_before_after = true
```

The serialized counter pack names the dominant measured sidecar cost:

```text
selector_dispatch_calls = 42336
selector_selected_calls = 216
selector_declined_calls = 42120
selector_error_calls = 0
input_materialization_calls = 216
bias_materialization_calls = 216
packed_matvec_calls = 216
output_tensor_construction_calls = 216
packed_matvec_ns = 19093586100
```

The classification is therefore:

```text
next_target = packed_matvec_compute
runtime_promotion_recommended = false
default_runtime_changed = false
speedup_claim = false
```

SLM-CPU-077 does not enable packed Q8_0 by default, broaden execution beyond
the exact tensor, claim speedup, claim sustained throughput, start Q4/Q5
runtime support, or involve server, accelerator, Qwen3.5, or BitNet QK256 work.

## SLM-CPU-078 Packed Matvec Compute Target

SLM-CPU-078 is the next queued implementation target after the SLM-CPU-077
counter artifact. It is scoped to reducing or further classifying
`packed_matvec_compute` for the opt-in exact-tensor packed Q8_0 sidecar path.

Any SLM-CPU-078 runtime change must preserve the Qwen3 Q8_0 appliance oracle
before making even a bounded improvement claim:

```text
model SHA unchanged
strict GGUF tokenizer authority unchanged
prompt IDs unchanged
generated IDs unchanged
decoded text unchanged
selected CPU backend/kernel identity unchanged
dense hook identity unchanged
fallback_used = false
default runtime = eager_f32_candle
packed sidecar scope = layers.0.attention.q_proj.weight only
```

If no safe optimization lands, SLM-CPU-078 should emit a concrete blocker or
next-target artifact rather than weakening the receipt boundary. It must not
enable packed Q8_0 by default, broaden packed sidecar execution to all dense
tensors, claim sustained 8250U throughput, claim broad answer quality, start
Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256 paths.

## SLM-CPU-079 Post-Aligned Matvec Artifact

SLM-CPU-079 captures the first real i5-8250U Qwen3 Q8_0 warm-session artifact
after the SLM-CPU-078 aligned packed Q8_0 exact-tensor matvec implementation.
The committed artifact pack is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-079-post-aligned-matvec-artifact.json
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-079-post-aligned-matvec-classification.json
```

The run keeps the packed sidecar path opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight` / `blk.0.attn_q.weight`. It records
`selected_path = packed_q8_sidecar`, `selected_kernel =
dense-q8-sidecar-linear`, strict GGUF tokenizer authority, `selected_backend =
cpu-rust`, and `fallback_used = false`.

The behavior oracle passes against the SLM-CPU-077 post-bridge sidecar oracle:

```text
prompt_ids_match = true
generated_ids_match = true
decoded_text_match = true
model_sha_match = true
tokenizer_source_match = true
tokenizer_strict_match = true
selected_backend_match = true
fallback_false_before_after = true
```

The post-aligned counter pack records:

```text
selector_dispatch_calls = 42336
selector_selected_calls = 216
selector_declined_calls = 42120
selector_error_calls = 0
input_materialization_calls = 216
bias_materialization_calls = 216
packed_matvec_calls = 216
output_tensor_construction_calls = 216
packed_matvec_ns = 16289575900
```

Compared with the SLM-CPU-077 post-bridge counter pack
(`packed_matvec_ns = 19093586100`), the classification is:

```text
result = improved_bounded_packed_matvec_counter
delta_packed_matvec_ns = -2804010200
packed_matvec_reduction_percent = 14.68561319656971
runtime_promotion_recommended = false
default_runtime_changed = false
speedup_claim = false
```

This is a bounded counter-level classification only. SLM-CPU-079 does not
claim end-to-end speedup, sustained 8250U throughput, broad answer quality,
Q4/Q5 runtime support, server execution, accelerator execution, Qwen3.5
support, or BitNet QK256 changes.

## Current Next Target

The current performance lane has two valid next-target families. Neither is a
default-runtime promotion gate by itself.

1. Continue the allocation path from the SLM-CPU-035 prefill attribution and
   the SLM-CPU-038 typed transformer-forward workspace boundary by removing or
   narrowly classifying one `prompt_prefill.forward` owned-output allocation.
2. Continue the packed Q8_0 exact-tensor path from SLM-CPU-079 by reducing the
   `packed_matvec_compute` counter or by collecting repeated before/after
   warm-session timing receipts, while keeping `packed_q8_sidecar` opt-in and
   exact-tensor scoped.

Both paths must use the Qwen3 Q8_0 appliance profile as the behavior oracle.
Any before/after artifact must preserve:

```text
model SHA
strict GGUF tokenizer authority
prompt IDs
generated IDs
decoded text
selected CPU backend/kernel identity
fallback_used = false
```

If a change improves a counter but changes generated IDs, decoded text, model
identity, tokenizer source, backend/kernel identity, or fallback state, the
change is a failed performance slice. If it helps only one model or one exact
tensor, the dashboard must say so.

This dashboard does not claim end-to-end speedup, sustained 8250U throughput,
broad answer quality, Q4/Q5 runtime support, server execution, GPU/NPU/OpenVINO
or UHD 620 execution, Qwen3.5 support, or BitNet QK256/I2_S changes.

## SLM-CPU-081 Repeated Timing Gate

SLM-CPU-081 records the next evidence boundary for the exact-tensor packed-Q8
path after the SLM-CPU-079 counter improvement:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-081-repeated-packed-q8-timing-gate.json
classification.result = not_claimed
minimum_receipts_per_side = 3
current_baseline_receipts = 1
current_candidate_receipts = 1
```

The available evidence still matters: the SLM-CPU-079 candidate preserved the
Qwen3 Q8_0 behavior oracle and reduced the bounded `packed_matvec_ns` counter
relative to the SLM-CPU-077 sidecar oracle. That is a counter-level result, not
repeated end-to-end timing evidence. A future timing classification must compare
at least three behavior-equivalent baseline receipts and three
behavior-equivalent candidate receipts for the same host, model SHA, thread
count, prompt corpus, tokenizer authority, backend identity, dense hook identity,
and `fallback_used = false` boundary.

The exact-tensor packed-Q8 sidecar remains opt-in, scoped to
`layers.0.attention.q_proj.weight`, and is not promoted to the default runtime by
this gate.
