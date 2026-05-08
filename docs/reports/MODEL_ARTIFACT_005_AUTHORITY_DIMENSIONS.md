# MODEL-ARTIFACT-005 Authority Dimensions

**Status:** diagnostic-only; no artifact is promoted to `answer_ready`

## Summary

`MODEL-ARTIFACT-005` splits artifact authority into explicit dimensions so
backend lanes can tell the difference between the official CUDA target and
alternate-quant control evidence.

The shared gate remains closed:

- No artifact is `answer_ready`.
- The official Microsoft I2_S GGUF remains the official RTX 5070 Ti CUDA target.
- The official Microsoft I2_S GGUF still fails the prompt suite under
  `ik_llama.cpp`.
- `tdh111` IQ2_BN_R4 passes the tiny prompt suite under its intended
  `ik_llama.cpp` runner, but it is an alternate quantization and still lacks
  required pre-tokenizer authority.

## Authority Fields

Artifact manifests now distinguish:

| Field | Purpose |
|---|---|
| `answer_readiness_scope` | Whether the artifact is the official target, an alternate-quant control, or diagnostic-only. |
| `target_alignment` | Whether the artifact aligns with the official I2_S CUDA target. |
| `runner_authority` | Which runner produced the relevant evidence. |
| `tokenizer_authority` | Whether tokenizer authority is present, missing, defaulted, or externally supplied. |
| `pretokenizer_authority` | Whether pre-tokenizer authority is present, missing, defaulted, or externally supplied. |
| `prompt_suite_result` | Whether the deterministic prompt suite passed, failed, was blocked, or was not run. |
| `can_unblock_official_i2s_cuda` | Whether the artifact can unblock the official RTX 5070 Ti I2_S CUDA answer lane. |
| `can_unblock_alt_quant_control` | Whether the artifact can support a separate alternate-quant answer-control lane. |

## Artifact Decisions

| Artifact | Scope | Target alignment | Runner | Prompt suite | Tokenizer/pre-tokenizer authority | Can unblock official I2_S CUDA? | Can unblock alt-quant control? |
|---|---|---|---|---|---|---|---|
| `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` | `official_target` | `official_i2s_cuda_target` | `ik_llama_cpp` | `failed` | missing | no | no |
| `tdh111_bitnet_b158_2b_4t_iq2_bn_r4` | `alternate_quant_control` | `official_derived_alt_quant` | `ik_llama_cpp` | `passed` | missing | no | yes |

## Implication

The `tdh111` result is useful because it proves an official-derived BitNet
artifact can produce readable tiny-suite output under its intended runner. It
does not unblock the official RTX 5070 Ti CUDA answer target because that lane
targets Microsoft I2_S, not IQ2_BN_R4, and the required pre-tokenizer authority
is still missing.

The next unblocker is `MODEL-ARTIFACT-006`: audit tokenizer and pre-tokenizer
authority for the official target and the alternate-quant control candidate.

## Claim Boundary

This report does not claim:

- coherent Rust CPU answers;
- coherent Rust CUDA answers;
- that `tdh111` satisfies the official Microsoft I2_S CUDA target;
- that any artifact is `answer_ready`;
- any speedup or performance result.
