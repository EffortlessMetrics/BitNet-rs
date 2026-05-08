# BITNET-COMPAT-001 Model/Kernel Compatibility

## Summary

`1bitLLM/bitnet_b1_58-3B` on x86 with `I2_S` is an upstream-unsupported
combination. It must not be used as answer-ready, reference-authority,
backend-parity, or speedup evidence for BitNet-rs CPU, AVX-512, or CUDA lanes.

The official Microsoft `BitNet-b1.58-2B-4T` I2_S artifact remains the x86
reference target for BitNet-rs backend answer gates.

## Source

The upstream bitnet.cpp README lists `microsoft/BitNet-b1.58-2B-4T` under
Official Models with x86 `I2_S` support. The same README lists
`1bitLLM/bitnet_b1_58-3B` under Supported Models with x86 `I2_S` unsupported,
x86 `TL2` supported, ARM `I2_S` unsupported, and ARM `TL1` supported.

The README's shown `setup_env.py` help lists `--quant-type {i2_s,tl1}`. Because
that help surface does not show `TL2`, this report records the 3B x86 `TL2`
route as `listed_supported_verify_runner` rather than proof authority.

Source: <https://github.com/microsoft/BitNet/blob/main/README.md>

## Compatibility Decisions

| Model | Role | Arch | Kernel | Status | Claim boundary |
|---|---|---|---|---|---|
| `microsoft/BitNet-b1.58-2B-4T` | official model | x86 | `I2_S` | `supported_reference` | May serve as the x86 reference path when paired with the existing answer-artifact, tokenizer, prompt, and backend receipt gates. |
| `microsoft/BitNet-b1.58-2B-4T` | official model | x86 | `TL2` | `supported` | Supported upstream, but not the current BitNet-rs official I2_S CUDA target. |
| `1bitLLM/bitnet_b1_58-3B` | supported model | x86 | `I2_S` | `unsupported_upstream` | Diagnostic-only. Must not be answer, reference, parity, or speed authority. |
| `1bitLLM/bitnet_b1_58-3B` | supported model | x86 | `TL2` | `listed_supported_verify_runner` | Listed upstream, but runner path still needs verification before proof claims. |
| `1bitLLM/bitnet_b1_58-3B` | supported model | ARM | `TL1` | `listed_supported_verify_runner` | Listed upstream, but runner path still needs verification before proof claims. |

## Validator Policy

The compatibility validator rejects unsupported or unverified model/kernel paths
for these proof claims:

- `answer_ready`
- `reference_authority`
- `backend_parity`
- `speedup`

Unsupported paths remain allowed for:

- `diagnostic_run`
- `artifact_inspection`
- `unsupported_path_receipt`

## Claim Boundary

This PR does not change QK256 math, tokenizer behavior, model loading,
transformer behavior, CUDA kernels, or server behavior. It records an authority
boundary so unsupported upstream model/kernel combinations cannot contaminate
CPU, AVX-512, CUDA, or benchmark proof surfaces.
