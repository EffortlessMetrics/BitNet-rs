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
