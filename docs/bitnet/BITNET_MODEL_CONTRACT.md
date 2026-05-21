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

## BitNet-Family Contract Matrix

BitNet-rs treats each BitNet-family artifact as a separate contract. A route is
not proof-authoritative just because a related model can load, a backend can run,
or another quantization variant can answer.

| Contract | Artifact / kernel | Current authority | Required next proof before stronger claims |
|---|---|---|---|
| Official Microsoft 2B I2_S | GGUF `I2_S` / QK256 | x86 CPU and RTX 5070 Ti CUDA reference lane with external `llama-bpe` tokenizer authority and `bitnetcpp-answer` prompt authority. The Intel Arc A770 OpenCL QK256 route is diagnostic route-receipt-only until model, quality, residency, and benchmark gates promote a stronger claim. | Profile-specific benchmark receipts before any speedup claim; full residency receipts before full CUDA residency claims; A770 stays diagnostic until its route receipts are paired with model, quality, residency, and benchmark proof. |
| Official Microsoft 2B TL1 | GGUF / TL1 LUT | Upstream-supported ARM lane, not a BitNet-rs answer authority yet. | TL1 parser, fixture parity, tokenizer/prompt authority, answer corpus, and ARM/NEON or Apple receipts. |
| Official Microsoft 2B TL2 | GGUF / TL2 LUT | Upstream-supported x86 alternate, not the current I2_S/QK256 CUDA target. | TL2 parser, AVX fixture parity, tokenizer/prompt authority, answer corpus, and benchmark receipts. |
| `1bitLLM/bitnet_b1_58-3B` x86 I2_S | GGUF / I2_S | Upstream-unsupported. | Diagnostic and unsupported-path receipts only. Must not become answer, reference, parity, or speed authority. |
| `1bitLLM/bitnet_b1_58-3B` x86 TL2 | GGUF / TL2 LUT | Listed upstream, runner path unverified. | Runner-path verification, tokenizer/prompt authority, answer corpus, and backend parity before proof claims. |
| `1bitLLM/bitnet_b1_58-3B` ARM TL1 | GGUF / TL1 LUT | Listed upstream, runner path unverified. | Runner-path verification, tokenizer/prompt authority, answer corpus, and backend parity before proof claims. |
| `tdh111` IQ2_BN_R4 | GGUF / alternate quant control | Useful alternate-control evidence only. | Cannot unblock the official Microsoft I2_S CUDA target; would need a separately scoped alternate-control lane. |

The machine-readable copy of this matrix lives in
`ci/model-artifacts/bitnet-model-contracts.toml`, with a Rust registry in
`crates/bitnet-models/src/model_contracts.rs`.

The user-facing cache verifier exposes the current reference contract:

```powershell
bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s --json
```

That command verifies the exact official GGUF bytes and includes the
`microsoft_bitnet_b158_2b_4t_i2s` contract summary in its JSON output. It does
not by itself prove CPU, CUDA, speedup, or full-residency claims; those still
require the strict backend receipts listed by the contract.

The full BitNet-family contract matrix is available through the model command:

```powershell
bitnet model contracts --json
bitnet model contracts onebitllm_bitnet_b158_3b_i2s_x86 --json
```

Known contracts that do not yet have a supported artifact identity and SHA256
registered for `bitnet model verify` fail closed with their contract summary.
For example, the upstream-unsupported 3B x86 `I2_S` contract may be inspected
or recorded as an unsupported-path diagnostic, but it cannot become answer,
reference, backend-parity, benchmark, or speed authority.

Contract summaries include architecture support rows so callers can distinguish
supported reference routes, proof-required routes, listed-but-unverified routes,
and upstream-unsupported routes before choosing a CPU or accelerator path.

## Dense Model Capability Summaries

Dense SLM artifacts use a separate `model_capability` verifier surface rather
than the BitNet-family `model_contract` surface. This keeps dense Qwen cache
metadata useful without letting dense regular-LLM evidence satisfy BitNet
packed QK256 proof.

The current Qwen2.5 0.5B cached artifacts are classified as dense SLM GGUF
artifacts:

| Artifact | Capability boundary |
|---|---|
| Qwen2.5 0.5B Instruct Q8_0 | Apple M4 CPU/NEON SLM answer lane metadata. Does not prove dense CUDA, BitNet QK256, server, speedup, or full-residency claims. |
| Qwen2.5 0.5B Instruct Q4_K_M | Storage/reference artifact only. Strict Rust execution remains unsupported. |

`bitnet model verify <qwen-id> --json` includes `model_capability` when the
artifact is recognized. The verifier still checks bytes and SHA256; the
capability summary only records the model family, artifact class, tokenizer and
prompt authority, known route boundary, permitted claims, and required receipts.

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
