# BITNET-SPEC-B158-3B-CONVERSION

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B artifact contract](BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md), [3B TL layout](BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; conversion/runner authority contract only
Policy impact: no policy exception

## Purpose

Define allowed conversion and runner-authority lanes for the 3B artifact. A
conversion receipt may prove a candidate artifact or an explicit blocker, but it
cannot by itself prove answer readiness, backend readiness, or speed.

## Allowed lanes

| Lane | Meaning | Claim |
| --- | --- | --- |
| `hf_safetensors_inventory` | Hugging Face shard inspection only. | No inference. |
| `transformers_reference` | Transformers, vLLM, or SGLang reference run on safetensors. | Reference only, not BitNet-rs. |
| `bitnetcpp_tl1_conversion` | Upstream-compatible ARM TL1 conversion. | Candidate TL1 artifact. |
| `bitnetcpp_tl2_conversion` | Upstream-compatible x86 TL2 conversion. | Candidate TL2 artifact. |
| `st2gguf_f16_reference` | BitNet-rs F16 structural conversion. | Structural/reference only. |
| `third_party_gguf` | External GGUF. | Diagnostic unless explicitly approved. |

## Required conversion proof

A conversion or blocked-conversion receipt must record:

- tool name, repository, and commit;
- exact command and host platform;
- input artifact hashes;
- output path class, output size, and output SHA256 when produced;
- output GGUF metadata needed to identify model family and quantization route;
- quantization route, either `tl1`, `tl2`, `f16_reference`, or
  `diagnostic_unknown`;
- whether tokenizer and pre-tokenizer authority are embedded or external;
- reference runner command and whether the runner loads the output;
- `claim_boundary = "diagnostic_only"` until reference quality passes.

## Blocked conversion receipts

A blocked route is useful evidence when it records the exact blocker:

```json
{
  "status": "blocked",
  "reason": "bitnet.cpp setup_env help exposes i2_s/tl1 only; TL2 path listed in support table but no reproducible command verified",
  "claim_boundary": "diagnostic_only"
}
```

## Hard rules

- The existing `bitnet-st2gguf` F16 path cannot satisfy TL1/TL2 packed BitNet
  proof.
- A reference runner load does not prove Rust CPU, CUDA, Apple, server, or speed
  readiness.
- x86 TL2 and ARM TL1 remain `listed_supported_verify_runner` until conversion
  and runner receipts identify reproducible commands.
- `I2_S`/QK256 conversion for this model is unsupported except diagnostic
  rejection receipts.
