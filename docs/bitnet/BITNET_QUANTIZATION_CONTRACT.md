# BitNet Quantization Contract

## Purpose

BitNet b1.58 proof is not generic INT8 inference proof. Receipts and benchmark artifacts must identify the BitNet quantization semantics before making correctness, kernel, or performance claims.

## Canonical Semantics

```yaml
bitnet_b1_58:
  weights:
    domain:
      - -1
      - 0
      - 1
    method: absmean
    bits_effective: 1.58
    note: trained native, not post-training quantized

  activations:
    precision: int8
    method: absmax
    granularity: per_token

  deployment:
    first_target_format: gguf_i2_s
    first_target_file: ggml-model-i2_s.gguf
    first_target_role: packed runtime proof

  reference:
    bf16_master_weights: reference, training, and fine-tuning material
```

## Required Receipt Fields

Every BitNet receipt must record:

```json
{
  "bitnet": {
    "weight_quantization": "W1.58",
    "activation_quantization": "A8",
    "weight_domain": "ternary",
    "weight_quantization_method": "absmean",
    "activation_quantization_method": "absmax",
    "activation_quantization_granularity": "per_token"
  }
}
```

## Proof Rules

- Do not treat BitNet as dense FP16 or FP32 matmul.
- Do not treat W1.58A8 as generic INT8 quantization.
- Do not claim packed-kernel performance from BF16 weights.
- Do not claim packed-kernel performance from OpenVINO graph smoke.
- Do not claim a native packed BitNet kernel unless the receipt names the consumed packed layout.
- If a path dequantizes before compute, the receipt must say so.

## Related Docs

- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
