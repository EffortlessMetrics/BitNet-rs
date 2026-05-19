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
| Prompt token cache | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json` | Validates that repeated rendered prompts reuse token IDs while preserving generated IDs/text and strict provenance |

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
kv_cache_recreated_per_prompt = true
sampler_recreated_per_prompt = false
sampler_reused_across_prompts = true
logits_buffer_reuse_claimed = false
```

The next safe optimization slices should start from these known remaining costs:

1. Reuse or isolate KV-cache buffers without changing prompt independence.
2. Continue reducing `model.logits` tensor allocation and output-head costs.
   SLM-CPU-026 removes fresh full logits Vec allocation from default
   repetition-penalty decode steps by reusing a host scratch buffer, but the
   model still produces logits tensors per token.
3. Continue keeping sampler and stop policy work out of the per-token hot loop
   where doing so preserves deterministic generated IDs. SLM-CPU-029 reuses one
   sampler across prompts for the temperature-zero Qwen3 profile only; nonzero
   temperature modes still recreate samplers to avoid RNG-state coupling.
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

SLM-CPU-052 is that next selector-update gate. It must start from the
SLM-CPU-051 readiness artifact and either keep the packed sidecar path blocked
with a concrete blocker, or update the selector only where the eager F32 Candle
oracle and the packed Q8_0 candidate preserve the same prompt IDs, generated
IDs, decoded text, strict tokenizer authority, selected CPU backend/kernel,
model SHA, and `fallback=false`. Any timing statement remains bounded to the
before/after receipts for the fixed Qwen3 Q8_0 appliance profile; it is not a
sustained-throughput or portable CPU-performance claim.

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
