# BITNET-SPEC-INTEL-GPU-RESIDENCY

## Purpose

Stop partial acceleration from becoming an accidental "full GPU" or "full
residency" claim.

## Residency classes

- `none`
- `kernel_smoke`
- `qk256_linears_only`
- `bitnet_trusted_partial`
- `dense_graph_runtime`
- `support_ops_partial`
- `decode_full`
- `full_device_resident`

## Phase table

Receipts should include phase residency as `gpu`, `cpu`, `mixed`,
`host_logits_only`, or `unknown` for weights, QK256 linears, dense linears,
embedding, LM head, KV cache, RMSNorm, RoPE, attention scores, softmax,
attention value mix, and sampling.

## Hard rules

QK256 linears on A770 are trusted partial acceleration, not full residency.
OpenVINO GPU LLMPipeline output is dense graph/runtime proof, not native OpenCL
residency.
