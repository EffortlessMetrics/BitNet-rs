# Falcon-E Family Campaign

Campaign ID: `falcon-e-family`

Status: active

## Objective

Register Falcon-E Family as BitNet-rs's compact direct-GGUF 1.58-bit model
family lane without overclaiming. The campaign moves from source-of-truth docs
to artifact identity, tokenizer/prompt authority, I2_S layout proof, reference
quality, CPU answers, backend receipts, benchmarks, CLI promotion, and exact
server profiles.

## Why This Exists

Falcon-E gives BitNet-rs a small secondary BitNet-like family with direct
`I2_S` GGUF artifacts for 1B and 3B instruct models. It is useful for fast
artifact onboarding and backend pressure tests, but it must not inherit proof
from Microsoft BitNet 2B, Falcon3, 1bitLLM, dense Falcon, or dense SLM lanes.

## End State

```text
For Falcon-E-1B and Falcon-E-3B, this exact GGUF artifact,
with this SHA256, tokenizer authority, prompt template, and stop policy,
loads through Rust, uses the correct I2_S route, produces coherent bounded
answers, passes strict CPU reference receipts, then graduates backend-by-backend
through AVX2/AVX512, CUDA, Apple CPU/NEON, and A770/OpenCL where receipts prove
fallback=false.
```

## Work Items

| Work item | Status | Notes |
|---|---|---|
| FE-000 | ready | Add source-of-truth docs, specs, and registered-only matrix rows. |
| FE-001 | proposed | Add exact 1B/3B artifact inventory receipts. |

## Claim Boundary

Do not claim:

```text
Falcon-E answer readiness
Falcon-E CPU/CUDA/Apple/A770 readiness
Falcon-E speedup
Falcon-E server readiness
Falcon-E full residency
Falcon-E 3B proof from Falcon-E 1B proof
Falcon-E proof inherited from Microsoft BitNet, Falcon3, 1bitLLM, or dense SLMs
I2_S/QK256 compatibility before Falcon-E layout proof
TL1/TL2 support before TL layout specs and scalar oracles
```

Do claim only:

```text
Falcon-E is registered as a compact 1.58-bit planning lane with direct I2_S GGUF
candidates, until later receipts prove narrower support levels.
```
