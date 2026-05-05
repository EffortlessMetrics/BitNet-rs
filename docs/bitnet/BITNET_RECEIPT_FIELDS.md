# BitNet Receipt Fields

## Purpose

Hardware receipts answer which machine/runtime/device ran. BitNet receipts must also answer which model, tokenizer, quantization format, kernel family, execution phase, and reference path ran.

## Minimum Fields

```json
{
  "model": {
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "sha256": "...",
    "format": "gguf",
    "architecture": "bitnet_b1_58",
    "context_length": 4096,
    "tokenizer": "llama3",
    "vocab_size": 128256,
    "loader_mode": "strict"
  },
  "bitnet": {
    "weight_quantization": "W1.58",
    "activation_quantization": "A8",
    "weight_domain": "ternary",
    "weight_quantization_method": "absmean",
    "activation_quantization_method": "absmax",
    "activation_quantization_granularity": "per_token",
    "kernel_format": "i2_s",
    "kernel_family": "i2_s|tl1|tl2|qk256|openvino_graph",
    "layout": "...",
    "layout_source": "gguf|converted|repo_internal",
    "fallback_layout": null
  },
  "execution": {
    "phase": "load_model|tokenize_prompt|prefill|first_token|decode_steady_state|sampling|total_generation|full",
    "prompt_tokens": 256,
    "generated_tokens": 128,
    "batch_size": 1,
    "thread_count": 4,
    "requested_backend": "...",
    "selected_backend": "...",
    "fallback_used": false
  }
}
```

## Required Hardware Linkage

BitNet receipt fields must be combined with hardware identity fields:

```json
{
  "requested_backend": "intel-i5-8250u-cpu-avx2",
  "selected_backend": "intel-i5-8250u-cpu-avx2",
  "runtime_api": "cpu",
  "resolved_device": {
    "name": "Intel Core i5-8250U"
  },
  "artifact_path": "ci/hardware/intel-i5-8250u/2026-05-05/strict-bitnet-proof.json"
}
```

## Loader Fields

Strict receipts must record:

```json
{
  "loader": {
    "mode": "real_gguf",
    "minimal_loader_fallback_used": false,
    "tokenizer_source": "model_artifact",
    "mock_tensors_used": false
  }
}
```

## Kernel Fields

Kernel receipts must record:

```json
{
  "kernel": {
    "family": "i2_s|tl1|tl2|qk256|openvino_graph",
    "implementation": "scalar|avx2|avx512|neon|metal|opencl|cuda|openvino",
    "layout": "...",
    "dequantizes_before_compute": false,
    "kernel_id": "..."
  }
}
```

## Reference Fields

Parity receipts must record:

```json
{
  "reference": {
    "name": "bitnet-rs-scalar|bitnet.cpp|cpu-reference",
    "artifact_path": "...",
    "max_abs_error": null,
    "mean_abs_error": null,
    "token_agreement": null
  }
}
```

## Claim Rules

- Missing model hash means the receipt cannot support reproducibility claims.
- Missing kernel family means the receipt cannot support kernel proof.
- Missing fallback status means the receipt cannot support hardware or BitNet proof.
- Missing reference path means parity cannot be claimed.
- Missing benchmark phase fields means performance cannot be claimed.

## Related Docs

- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
