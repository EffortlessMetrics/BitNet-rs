# bitnet_b1_58-large source map

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact](../../specs/BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [conversion](../../specs/BITNET-SPEC-B158-LARGE-CONVERSION.md), [tokenizer/prompt](../../specs/BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md), [reference quality](../../specs/BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [CPU](../../specs/BITNET-SPEC-B158-LARGE-CPU.md), [CUDA](../../specs/BITNET-SPEC-B158-LARGE-CUDA.md), [Apple](../../specs/BITNET-SPEC-B158-LARGE-APPLE.md), [performance](../../specs/BITNET-SPEC-B158-LARGE-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [plans/bitnet-b158-large](../../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion
Policy impact: no policy exception

## Current authority snapshot

| Dimension | Current truth | Claim impact |
| --- | --- | --- |
| Source repo | `1bitLLM/bitnet_b1_58-large` | Candidate source only. |
| Upstream status | bitnet.cpp lists the model as a supported 0.7B BitNet model. | Does not prove BitNet-rs answer support. |
| Upstream routes | x86 `I2_S`/`TL2`; ARM `I2_S`/`TL1`. | Each route still needs BitNet-rs runner-path receipts. |
| HF file shape | `model.safetensors`, tokenizer/config files, and Python assets. | No official GGUF claim from the inspected listing. |
| Recorded revision | `85d047191dcb224f0e04f20d26110caaf8dc1a47` in the Apple candidate matrix. | Blocks answer/backend claims until inventory and conversion authority are added. |
| Existing converter | `bitnet-st2gguf` can produce F16 GGUF structural/reference output. | Not packed `I2_S`, `TL1`, or `TL2` proof. |

## End-state chain

```text
artifact inventory
→ conversion authority
→ tokenizer/prompt authority
→ reference-good output
→ CPU answer-ready
→ accelerator answer-ready
→ exact-profile benchmark
→ product CLI support
→ exact-profile server support
```

Every promotion must be receipt-backed and fallback-explicit. A later backend
receipt may cite this source map only as planning context; it must cite exact
artifact, tokenizer, prompt, runner, backend, generated-token, and timing
receipts for support claims.

## Artifact authority checklist

Before any artifact claim, record:

- source revision;
- complete file list;
- file sizes;
- SHA256 for every required file;
- tokenizer files and hashes;
- relevant config fields;
- storage context;
- cleanup status;
- claim boundary with answer, backend, and speedup booleans set to false until
  later gates pass.

## Conversion authority checklist

Allowed lanes are:

- `hf_safetensors_structural` for source inspection only;
- `st2gguf_f16_reference` for structural/reference GGUF only;
- `bitnetcpp_i2s_conversion` for upstream-compatible `I2_S` after proof;
- `bitnetcpp_tl1_conversion` for upstream-compatible `TL1` after proof;
- `bitnetcpp_tl2_conversion` for upstream-compatible `TL2` after proof;
- `third_party_gguf` for diagnostics unless an explicit artifact-authority
  decision approves it.

## Claim boundaries

Do not claim:

- coherent local answers;
- CPU answer readiness;
- CUDA answer readiness;
- Apple local-answer success;
- Metal, OpenCL, NPU, or server readiness;
- performance or speedup;
- official Microsoft 2B proof inheritance;
- dense SLM proof inheritance.
