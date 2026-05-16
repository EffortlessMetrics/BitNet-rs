# SmolLM2 360M Strict CPU Sanity

## Scope

`SLM-CPU-017` starts the second small dense GGUF CPU sanity slice after the
Qwen3 0.6B Q8_0 appliance profile. The selected candidate is the pinned
SmolLM2 360M Instruct Q8_0 artifact from the CUDA productization ladder.

This slice records a strict CPU preflight blocker. It does not claim SmolLM2
CPU answer readiness, broad answer quality, sustained throughput, CUDA, server
readiness, OpenVINO, NPU, UHD 620, Qwen3.5 support, Q4/Q5 expansion, or BitNet
QK256 behavior.

## Candidate

| Field | Value |
|---|---|
| Model id | `smollm2-360m-instruct` |
| Artifact id | `smollm2-360m-instruct-q8_0` |
| Contract id | `smollm2_360m_instruct_q8_0` |
| Repository | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF` |
| Revision | `593b5a2e04c8f3e4ee880263f93e0bd2901ad47f` |
| File | `smollm2-360m-instruct-q8_0.gguf` |
| SHA256 | `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201` |
| Bytes | `386404992` |
| GGUF architecture | `llama` |
| Quantization | `Q8_0` |
| Tokenizer source | `gguf_metadata` |
| Pre-tokenizer | `smollm` |
| Prompt template | `smollm-chat` |

The local downloaded artifact used for the run matched the contract SHA256 and
byte count before execution.

## Attempt

The strict CPU preflight used the current `bitnet run` command surface with the
same claim discipline as the Qwen3 SLM CPU receipts:

```powershell
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  --device cpu `
  run `
  --model C:\bntarget\slm-cpu-017-models\smollm2-360m-instruct-q8_0.gguf `
  --prompt-template smollm-chat `
  --system-prompt "You are a helpful assistant." `
  --prompt "What is 2+2? Answer with only the number." `
  --max-new-tokens 1 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --logits-dump-steps 1 `
  --logits-topk 10 `
  --json-out C:\bntarget\slm-cpu-017-receipts\smollm2-strict-cpu-run.json `
  --no-warnings
```

The command selected the strict CPU backend before model load:

```text
requested_backend = cpu
selected_backend = cpu-rust
runtime_api = cpu
fallback_used = false
```

## Result

The run failed before tokenizer loading, prompt rendering, or generation:

```text
status = blocked_before_inference
stage = strict_gguf_load
failure_class = strict_loader_layernorm_gamma_guard
failed_tensor = blk.0.ffn_norm.weight
observed_rms = 0.09831
```

The strict loader rejected the artifact with:

```text
LayerNorm gamma 'blk.0.ffn_norm.weight' suspicious: rms=0.09831
```

Because generation did not start, there are no prompt token IDs, generated token
IDs, decoded text, logits, quality pass, or throughput result to claim.

## Receipt

Machine-readable evidence:

```text
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-15/smollm2-360m-strict-cpu-preflight-blocker.json
```

The receipt records the exact artifact contract, selected CPU backend,
`fallback_used=false`, strict loader/tokenizer request, the failing tensor and
stage, null prompt/generated IDs because generation did not start, and explicit
false claim booleans.

## Baseline Boundary

This same-box 9950X3D preflight does not replace the Kaby Lake i5-8250U Qwen3
Q8_0 appliance profile. The Qwen3 profile remains the established SLM CPU
baseline. SmolLM2 remains a second-model candidate blocked before CPU answer
sanity until exact metadata-scoped SmolLM2 normalization validation is
implemented and strict CPU sanity is retried.

## Claim Boundary

This page may claim only that the pinned SmolLM2 360M artifact reached strict
CPU model-load preflight on the 9950X3D box and failed closed without fallback.
It must not claim SmolLM2 CPU answer quality, broad dense SLM support, sustained
throughput, CUDA, server readiness, OpenVINO, NPU, UHD 620, Qwen3.5 support,
Q4/Q5 expansion, BitNet QK256 behavior, or inherited proof from Qwen2.5, Qwen3,
or Apple M4 evidence.
