# Qwen3 Multi-Token Stability on i5-8250U

This note records the `SLM-CPU-010` evidence shape for bounded Qwen3-0.6B
Q8_0 multi-token stability on the i5-8250U strict CPU path.

## Scope

This is a correctness and determinism artifact, not a performance claim.

It uses:

- model: `Qwen/Qwen3-0.6B-GGUF`
- file: `Qwen3-0.6B-Q8_0.gguf`
- sha256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- prompt template: `qwen`
- Qwen thinking policy: `--no-think`
- backend: `cpu`
- selected backend: `cpu-rust`
- selected kernel/runtime: `dense-qwen-cpu-reference`
- fallback: `false`

## Corpus

The bounded stability corpus is:

```text
ci/quality/slm-multitoken-stability.yaml
```

It covers three fixed prompts with `max_new_tokens` caps of 4, 8, and 16.
Each prompt is run twice under deterministic greedy settings. Stability is
scored by exact repeated prompt ID and generated ID equality.

## Evidence

The i5-8250U evidence bundle is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-multitoken-stability-run1.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-multitoken-stability-run2.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-multitoken-stability-validation.json
```

The validation artifact records `passed=true` when:

- both runs pass the bounded quality gates
- prompt IDs match between repeated runs
- generated IDs match between repeated runs
- strict GGUF tokenizer metadata is recorded
- `fallback_used=false`

## KV Metadata

The current receipt surface records prompt-prefill and decode-start position
metadata, including `kv_cache_behavior=prompt_prefix_prefilled_before_decode`.
It does not yet expose per-token KV append/read positions. The validation
artifact records this limitation explicitly as
`kv_cache_position_metadata_scope=prompt_prefill_and_decode_start_only`.

## Claim Boundary

This evidence does not claim:

- broad Qwen answer quality
- warm-session usability
- sustained i5-8250U throughput
- Q4/Q5 quant support
- server, GPU, NPU, OpenVINO, or UHD 620 execution
- Qwen3.5 or hybrid Qwen support
- BitNet QK256/I2_S kernel changes
