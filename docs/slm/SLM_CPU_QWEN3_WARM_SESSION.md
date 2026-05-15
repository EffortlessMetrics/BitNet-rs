# Qwen3 CPU Warm-Session Receipt

This note records the SLM-CPU-011 evidence for the i5-8250U Kaby Lake lane.
The lane target is a slow, strict CPU proof host for small dense GGUF models,
not a throughput or server claim.

## Artifact

Command shape:

```powershell
$env:BITNET_STRICT_MODE = "1"
$env:BITNET_DISABLE_MINIMAL_LOADER = "1"
$env:RAYON_NUM_THREADS = "1"

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  --device cpu `
  slm-warm-session `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --corpus ci\quality\slm-warm-session-corpus.yaml `
  --corpus-repeat-runs 2 `
  --max-new-tokens 8 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --threads 1 `
  --strict-loader `
  --strict-tokenizer `
  --prompt-template qwen `
  --require-determinism `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-warm-session.json
```

Outputs:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-warm-session-validation.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-warm-session-prompts/
```

## Evidence

The aggregate receipt records:

```text
artifact_kind = slm_cpu_warm_session
model.sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
tokenizer.source = gguf_metadata
tokenizer.strict = true
requested_backend = cpu
selected_backend = cpu-rust
runtime_api = cpu
fallback_used = false
session.model_loaded_once = true
session.tokenizer_loaded_once = true
generation.qwen_no_think = true
quality_summary.passed = true
determinism.checked = true
determinism.passed = true
speedup_claim = false
```

The corpus runs three prompts twice each in one process. The repeated runs
produce identical generated token IDs and text for each case.

## Boundaries

This is warm-session operability evidence for Qwen3-0.6B Q8_0 on the local
i5-8250U CPU path. It does not claim sustained throughput, GPU, NPU, OpenVINO,
UHD 620, server inference, Qwen3.5 support, Q4/Q5 support, or BitNet QK256
kernel changes.

The receipt includes explicit `not_sampled_in_slm_cpu_warm_session` fields for
resident memory, thermal, and power context. Those fields are present so later
efficiency work can replace them with measured values without changing the
claim boundary.

## SLM-CPU-012 Cleanup Boundary

The next cleanup slice keeps the SLM-CPU-011 generated IDs and quality gates as
the behavior oracle. It may reduce prompt-loop allocation and layout waste, but
must keep the same strict CPU backend, GGUF tokenizer authority,
`fallback_used=false`, and Qwen no-thinking prompt policy.

The warm-session receipt now exposes bounded reuse evidence for the prompt
buffers and records that prompt-invariant stop sequences and stop token IDs are
computed once per session. This is allocation/layout evidence only; resident
memory, thermal, and power fields remain explicitly unavailable unless a later
hardware-specific sampler fills them in.

## SLM-CPU-013 Q8_0 Hot-Path Boundary

The first Q8_0 hot-path cleanup keeps the SLM-CPU-011/012 generated IDs and
strict receipt provenance as the behavior oracle. Dense GGUF linear layers that
omit a bias now use the no-bias `Linear` path instead of materializing a zero
bias tensor, preserving numerical output while avoiding unnecessary allocation
and per-forward bias addition. This is implementation cleanup only; it does not
claim sustained throughput, Q4/Q5 support, GPU/NPU/OpenVINO/UHD 620 execution,
server inference, Qwen3.5 support, or BitNet QK256 kernel changes.
