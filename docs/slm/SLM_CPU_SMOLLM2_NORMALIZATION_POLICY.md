# SmolLM2 360M Normalization Policy Audit

## Scope

`SLM-CPU-018` records the policy decision after `SLM-CPU-017` failed closed on
the pinned SmolLM2 360M Instruct Q8_0 artifact. This is an audit and decision
artifact only. It does not change the loader, retry CPU answer sanity, or start
CUDA planning.

The blocker receipt is:

```text
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-15/smollm2-360m-strict-cpu-preflight-blocker.json
```

The receipt records:

```text
stage = strict_gguf_load
failure_class = strict_loader_layernorm_gamma_guard
failed_tensor = blk.0.ffn_norm.weight
observed_rms = 0.09831
tokenizer_load_reached = false
generation_reached = false
fallback_used = false
```

## Evidence

The same pinned SmolLM2 artifact has three relevant evidence surfaces:

| Surface | Evidence | Meaning |
|---|---|---|
| Artifact contract | `docs/reports/SMOLLM2_360M_ARTIFACT_CONTRACT.md` and `ci/model-artifacts/dense-slm-model-contracts.toml` | The artifact identity, SHA256, GGUF metadata, tokenizer, prompt template, and license are known. |
| Reference runner | `ci/quality/apple-m4-slm-model-breadth-reference-sanity.toml` | A llama.cpp reference path produced plausible bounded SmolLM2 outputs for the same SHA. |
| Rust strict loader | `ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-15/smollm2-360m-strict-cpu-preflight-blocker.json` | BitNet-rs currently rejects the artifact before tokenizer, prompt rendering, or generation. |

The failing tensor is a normalization weight on a GGUF whose
`general.architecture` is `llama` and whose tokenizer pre-tokenizer is
`smollm`. Treating this as plain generic LLaMA is too broad for a product
exception, because it would relax validation for unrelated LLaMA-family GGUFs.

## Decision

Keep the generic strict LayerNorm/RMSNorm gamma guard fail-closed.

Record a governed SmolLM2-family exception path for a later implementation PR,
limited to exact model-family metadata. A future loader change may accept the
SmolLM2 normalization envelope only when the loaded artifact matches all of the
following:

- `general.architecture = "llama"`;
- tokenizer metadata identifies the `smollm` pre-tokenizer;
- model contract is `smollm2_360m_instruct_q8_0`;
- artifact SHA256 is
  `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`;
- tensor naming and dimensions match the SmolLM2 360M contract;
- the loader records the exception in the retry receipt.

The exception must not be implemented as a generic `llama` relaxation, an
environment fallback, or a non-strict warning downgrade.

## Next Implementation Boundary

`SLM-CPU-019` implements the exact SmolLM2 model-family normalization
validation boundary. The loader path remains fail-closed for generic `llama`
normalization weights, and the SmolLM2 exception is selected only from the exact
artifact SHA plus GGUF metadata and dimensions recorded in this audit.

`SLM-CPU-020` broadens the exact SmolLM2 model-family normalization envelope
only for the pinned artifact after the first retry showed layer-varying
attention norm RMS values. The generic `llama` guard remains fail-closed; the
exception still requires the exact artifact SHA plus GGUF metadata and
dimensions recorded in this audit.

The strict CPU sanity retry reached tokenizer authority, prompt rendering, and
one-token generation with `fallback_used=false`, but it generated `The` for the
math prompt. That is a diagnosable post-load quality failure, not CPU answer
readiness. SmolLM2 must not advance to all-layer CUDA planning until the
wrong-first-token behavior or prompt/template policy is resolved.

Required implementation checks:

- generic `llama` normalization weights outside the strict envelope still fail;
- exact SmolLM2 360M metadata can select the governed SmolLM2 envelope;
- the accepted exception is receipt-visible;
- CPU answer readiness remains false until generation evidence exists.

`SLM-CPU-019` implementation evidence:

```text
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-normalization-validation-implementation.json
```

`SLM-CPU-020` retry evidence:

```text
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-strict-cpu-sanity-retry.json
```

## Claim Boundary

This audit may claim only that the next SmolLM2 blocker is a governed
normalization-policy decision. It must not claim SmolLM2 CPU answer readiness,
broad dense SLM support, sustained throughput, CUDA, server readiness, OpenVINO,
NPU, UHD 620, Qwen3.5 support, Q4/Q5 expansion, BitNet QK256 behavior, or proof
inherited from Qwen2.5, Qwen3, Apple M4, or the reference runner.
