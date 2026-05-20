# BITNET-SPEC-LLAMA3-8B-158-CONVERSION

Status: proposed
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no support promotion
Policy impact: no policy exception

## Purpose

Define allowed conversion and reference-runner lanes for the Llama3 8B 1.58
candidate. Conversion evidence identifies candidate artifacts; it does not imply
answer quality or backend support until later specs pass.

## Allowed lanes

| Lane | Meaning | Claim |
| --- | --- | --- |
| `hf_safetensors_inventory` | HF file inventory only. | No inference claim. |
| `transformers_reference` | Transformers loads safetensors. | Reference only. |
| `vllm_reference` | vLLM serves safetensors. | Reference only. |
| `sglang_reference` | SGLang serves safetensors. | Reference only. |
| `st2gguf_f16_reference` | BitNet-rs F16 GGUF structural conversion. | Structural/reference only. |
| `bitnetcpp_i2s_conversion` | Upstream-compatible I2_S conversion. | Candidate packed I2_S artifact. |
| `bitnetcpp_tl1_conversion` | Upstream-compatible TL1 conversion. | Candidate ARM TL1 artifact. |
| `bitnetcpp_tl2_conversion` | Upstream-compatible TL2 conversion. | Candidate x86 TL2 artifact. |
| `third_party_gguf` | External GGUF. | Diagnostic unless approved. |

## Conversion proof fields

Each proof must record tool name, tool repo, tool commit, command, host
platform, input hashes, output file, output size, output SHA256, GGUF metadata,
quantization route (`i2_s`, `tl1`, `tl2`, or `f16`), runner command, whether the
tokenizer/pre-tokenizer is embedded or external, and whether a reference runner
loads it.

Blocked conversion is valid evidence when explicit:

```json
{
  "status": "blocked",
  "reason": "No approved I2_S/TL1/TL2 conversion output yet",
  "claim_boundary": "diagnostic_only"
}
```

## Hard rules

- An F16 structural GGUF is not packed `I2_S`, `TL1`, or `TL2` proof.
- A third-party GGUF is diagnostic unless a separate artifact-authority decision
  approves it.
- Conversion success does not prove reference-good output.
- Conversion success does not prove CPU, CUDA, Apple, server, or speed support.
