# BitNet Model Contract

## Purpose

Hardware proof is not enough for BitNet proof. Any artifact that claims BitNet progress must record the model, tokenizer, quantization format, kernel family, execution phase, reference path, and fallback status.

## Canonical Model

Initial canonical model contract:

```yaml
canonical_model:
  hf_repo: microsoft/bitnet-b1.58-2B-4T
  gguf_repo: microsoft/bitnet-b1.58-2B-4T-gguf
  deployment_variant: packed_1_58_bit
  reference_variant: bf16_master_weights
  gguf_file: ggml-model-i2_s.gguf
  architecture: bitnet_b1_58
  parameters: approx_2b
  training_tokens: 4t
  context_length: 4096
  tokenizer: llama3
  vocab_size: 128256
  quantization: W1.58A8
  weight_domain: ternary {-1, 0, +1}
  activation_quantization: int8 per-token absmax
```

## Architecture Contract

The canonical BitNet b1.58 2B4T proof contract includes:

- BitLinear layers.
- RoPE position encoding.
- ReLU2 FFN activation.
- `subln` normalization.
- No bias terms.
- W1.58A8 quantization.
- Ternary weights.
- Absmean weight quantization.
- Per-token 8-bit activation quantization.
- Context length 4096.
- LLaMA 3 tokenizer.
- Vocabulary size 128,256.

Model artifact types must be distinguished:

| Artifact | Use |
|---|---|
| Packed deployment weights | Deployment/kernel proof |
| BF16 master weights | Reference/model-shape work, not packed-kernel performance |
| GGUF weights | bitnet.cpp and bitnet-rs proof fixtures |

Hard rule:

```text
BF16 master weights are reference/training/fine-tuning material.
GGUF/I2_S packed weights are the first deployment/runtime proof target.
```

The first canonical GGUF fixture path is:

```text
models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

Do not add model files to docs PRs.

## Strict Proof Requirements

Strict BitNet proof requires:

- Real GGUF model.
- Real tokenizer artifact or tokenizer embedded in GGUF.
- No minimal loader fallback.
- Tokenizer loaded from a real model/tokenizer artifact.
- No mock tensors.
- No fake kernels.
- Selected kernel path recorded.
- Selected backend recorded.
- Fallback path recorded if any.
- `fallback_used=false`.
- Model file hash recorded when available.
- Hardware selected backend recorded separately from BitNet kernel family.

## Hard Rules

- Minimal GGUF fallback cannot support correctness claims.
- Minimal loader fallback may support compatibility testing only.
- BF16 master weights cannot support packed-kernel performance claims.
- OpenVINO graph conversion cannot support native BitNet kernel claims.
- CPU fallback cannot support GPU/NPU claims.
- Hardware smoke without model/kernel fields cannot support BitNet progress claims.

## Required Receipt Fields

Every BitNet proof artifact should include at least:

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
    "kernel_format": "i2_s",
    "kernel_family": "i2_s|tl1|tl2|qk256|openvino_graph",
    "layout": "...",
    "fallback_layout": null
  }
}
```

## Related Docs

- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_REFERENCE_RUNS.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/bitnet/BITNET_PARITY_TOLERANCES.md`
