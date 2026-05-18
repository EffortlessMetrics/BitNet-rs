# BITNET-SPEC-CPU-AVX2-HOTPATH: CPU AVX2 BitNet Hot-Path Contract

Status: proposed
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`,
`docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADR: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Linked plan: `plans/cpu-avx2-bitnet/implementation-plan.md`
Applies to: official Microsoft BitNet I2_S/QK256 CPU AVX2 proof, strict CPU
answer receipts, QK256 kernel selection, CPU AVX2 performance receipts, and
user-facing CPU AVX2 status claims.

## Target end state

AVX2 CPU is fully working only when the official BitNet I2_S/QK256 model runs
through the normal Rust CPU user path with strict GGUF loader authority, strict
tokenizer authority, correct intelligible answer receipts, selected AVX2 BitNet
kernels, no hidden scalar/dequantized fallback, stable scalar-vs-AVX2 generated
token parity, and exact-profile prefill/first-token/decode performance that has
been reviewed and promoted profile-by-profile.

## Requirements

The requirements below define the minimum behavior before this lane may promote
CPU AVX2 BitNet I2_S/QK256 execution.

## Strict fallback rules

- Strict requested AVX2 must fail if AVX2/FMA or any other required CPU feature
  is unavailable.
- Strict requested AVX2 must fail if the selected path is scalar, dequantized,
  diagnostic, mock, reference-only, or the no-scale F32 helper when inline-scale
  BitNet execution is required.
- `fallback_used=false` is invalid if hot-path counters show scalar
  substitution, dequantized execution, or missing AVX2 invocations for a
  selected AVX2 kernel.
- Non-strict fallback may select scalar only if the receipt records the selected
  scalar kernel and an explicit fallback reason.

## Required receipt fields

Every strict CPU AVX2 proof receipt must include these fields or their canonical
schema equivalents:

```json
{
  "requested_backend": "cpu",
  "selected_backend": "cpu-rust",
  "requested_kernel": "...",
  "selected_kernel": "...",
  "kernel_family": "i2_s|qk256",
  "runtime_api": "cpu",
  "fallback_used": false,
  "fallback_reason": null,
  "model": {
    "loader_mode": "real_gguf",
    "quant_format": "i2_s",
    "sha256": "..."
  },
  "tokenizer": {
    "source": "...",
    "strict": true
  },
  "qk256_hot_path": {
    "scaled_i8s_scalar_invocations": 0,
    "scaled_i8s_avx2_invocations": 0,
    "f32_scalar_invocations": 0,
    "f32_avx2_invocations": 0,
    "flat_bytes_extracted_count": 0,
    "input_rows_materialized_count": 0,
    "output_rows_allocated_count": 0,
    "tensor_to_vec_count": 0
  }
}
```

Performance receipts must additionally include phase timings for the profile
being claimed, including model load, tokenizer load, prompt rendering, prefill,
first token, decode total, tokens per second, workload shape, CPU feature set,
model identity, selected backend/kernel, and fallback status.

## Scalar parity gate

The canonical scalar packed path remains the correctness oracle. AVX2 kernel
work must compare directly against scalar packed results, and transformer-level
optimization work must preserve generated token IDs against scalar under greedy,
deterministic settings. Any generated-token drift requires a divergence receipt
and blocks optimization promotion until it is classified and accepted.

## Scaled I2_S x I8_S hot-path requirement

Inline-scale BitNet inference must use the scaled I2_S x I8_S flow: quantize
activations to I8_S, compute over packed I2_S codes, and apply the same scale
and correction semantics as the scalar `gemv_qk256_bitnet_i8s_scaled` oracle.
The no-scale F32 QK256 AVX2 GEMV is not a substitute for this path and must not
be counted as scaled BitNet AVX2 proof.

A selected scaled AVX2 kernel is valid only when the receipt records
`scaled_i8s_avx2_invocations > 0`, `scaled_i8s_scalar_invocations = 0` for the
strict AVX2 proof run, `fallback_used=false`, and a selected kernel ID that
names the scaled AVX2 path.

## Performance promotion requirements

Performance promotion is exact-profile only. Each promoted profile must have a
receipt-backed scalar comparator and AVX2 measurement for the same model
artifact, tokenizer authority, prompt policy, workload shape, and output rule.
Promotion decisions must state accepted/rejected and why for profiles such as
micro scaled GEMV, first token, `decode_128`, `prefill_128`, `prefill_512`, and
warm session profiles. A result for one profile does not imply another.

## Proof commands

Documentation-only changes for this spec run:

```bash
cargo fmt --all -- --check
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
git diff --check
```

Runtime changes must also run the scoped CPU AVX2 commands listed by their
campaign work item and emit or validate the receipts required by this spec.

## Claim boundary

A claim is valid only for the exact model artifact, tokenizer source, prompt
policy, backend, selected kernel, workload profile, and receipt named by the
proof. Adjacent BitNet, dense, accelerator, server, or quality families are not
promoted by implication.

## Non-goals and forbidden claims

CPU AVX2 BitNet hot-path proof must not claim any of the following unless a
separate authority and receipt family proves it:

- CUDA, NPU, OpenVINO, A770, Apple M4, or other accelerator support;
- dense SLM, Qwen, Llama, Gemma, Phi, or broad small-LLM support;
- server readiness, streaming, concurrency, or endpoint performance;
- global speedup without exact-profile review;
- broad chat quality beyond the governed answer corpus and artifact gate;
- support for all BitNet models from one official Microsoft I2_S/QK256 artifact.
