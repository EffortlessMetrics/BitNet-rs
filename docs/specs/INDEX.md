# Specs Index

## Common Local Inference Status Contracts

Use these specs before changing the machine-readable support surfaces for
model readiness, receipt explanation, or support bundles:

1. [Model Readiness Status Surface](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md)
   - Defines the stable `bitnet model status --format json` dashboard fields,
     tier semantics, readiness booleans, backend/route/fallback provenance, and
     proof-family booleans.
2. [Receipt Explain Schema](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md)
   - Defines the normalized `bitnet receipts explain --format json` shape,
     flat support aliases, nested diagnostic objects, unknown-vs-false
     semantics, and promotion warnings.
3. [Support Bundle](BITNET-SPEC-SUPPORT-BUNDLE.md)
   - Defines `bitnet support bundle --latest --device <device> --format json`
     as a read-only issue artifact that embeds model status and receipt
     explanation without promoting claims. The older
     [Support Bundle Schema](BITNET-SPEC-SUPPORT-BUNDLE-SCHEMA.md) URL remains
     as a compatibility pointer.
4. [CUDA Support Issue](BITNET-SPEC-CUDA-SUPPORT-ISSUE.md)
   - Defines the receipt-backed CUDA issue template contract, required support
     bundle field, JSON rendering, and claim-boundary checklist.
5. [Claim-Boundary Review](BITNET-SPEC-CLAIM-BOUNDARY-REVIEW.md)
   - Defines the cross-cutting PR review rule that diagnostics, route
     visibility, one-profile receipts, support bundles, and microbench evidence
     do not promote support claims without exact receipts.
6. [Proof-Family Non-Inheritance](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md)
   - Defines the common rule that proof does not transfer across model family,
     artifact, tokenizer/prompt authority, backend, route, hardware profile,
     server profile, benchmark profile, or residency class.

Do not remove or repurpose support JSON fields without a schema-version bump.
Do not let status, receipt explanation, or support bundles infer speedup,
residency, broad server readiness, or cross-family proof.

## PR Queue And CI Operations

Use these specs before closing, replacing, restacking, or otherwise writing to
PR queue state:

1. [PR Queue Disposition](BITNET-SPEC-PR-QUEUE-DISPOSITION.md)
   - Defines valid close reasons, invalid close reasons, routing states,
     successor requirements, and PR identity preservation.
2. [PR Write-Action CI Economics](BITNET-SPEC-PR-CI-ECONOMICS.md)
   - Defines queue write actions, no-CI-for-archaeology, bulk-write limits,
     rerun rules, and allowed CI spend during burn-down.
3. [Generated Tracking](BITNET-SPEC-GENERATED-TRACKING.md)
   - Defines generated dashboard ownership, conflict handling, source-first
     regeneration, and no hand-authored status edits.

Wrong base, stale stack, closed parent, and needs-restack are routing states,
not close reasons. Queue writes should follow content review and should spend
CI only on current merge candidates, clean ports, or required proof.
Generated tracking conflicts should be fixed in campaign manifests, events,
generators, or checkers before regenerating output.

## TL1 ARM-First Table-Lookup Route

Use these artifacts before implementing or claiming TL1 support:

1. [TL1 Source Map](../bitnet/tl1/README.md)
   - Defines TL1 as an ARM-first route with explicit non-inheritance boundaries from `I2_S`/QK256 and TL2.
2. [TL1 Implementation Plan](../../plans/tl1/implementation-plan.md)
   - Sequences docs/spec setup, layout/scalar proof, artifact authority, Apple CPU/NEON proof, and benchmark promotion.
3. [TL1 Campaign](../tracking/campaigns/tl1/CAMPAIGN.md)
   - Tracks the active work item and proof commands.

Do not treat upstream listed ARM TL1 support as answer/backend/speed proof. x86 TL1 remains `unsupported_upstream` for currently tracked families unless the compatibility ledger changes with new upstream evidence.

## I2_S/QK256 Productization

Use these artifacts before implementing or claiming new I2_S route capabilities:

1. [I2_S Source Map](../bitnet/i2s/README.md)
2. [I2_S Implementation Plan](../../plans/i2s/implementation-plan.md)
3. [I2_S Productization Proposal](../proposals/BITNET-PROP-0015-i2s-productization.md)
4. [I2_S Layout](BITNET-SPEC-I2S-QK256-LAYOUT.md)
5. [I2_S Scaled Math](BITNET-SPEC-I2S-SCALED-I8S-MATH.md)
6. [I2_S Kernel Identity](BITNET-SPEC-I2S-KERNEL-IDENTITY.md)
7. [I2_S Model Compatibility](BITNET-SPEC-I2S-MODEL-COMPATIBILITY.md)
8. [I2_S Artifact Gate](BITNET-SPEC-I2S-ARTIFACT-GATE.md)
9. [I2_S CPU](BITNET-SPEC-I2S-CPU.md), [CUDA](BITNET-SPEC-I2S-CUDA.md), [A770/OpenCL](BITNET-SPEC-I2S-OPENCL-A770.md), [Apple](BITNET-SPEC-I2S-APPLE-NEON-METAL.md)
10. [I2_S Performance](BITNET-SPEC-I2S-PERFORMANCE.md) and [Status Surface](BITNET-SPEC-I2S-STATUS-SURFACE.md)

Do not claim TL1/TL2, dense SLM, global speedup, or full residency from I2_S proof.

## Official Microsoft BitNet 2B Productization

Use these artifacts before implementing or claiming support for
`microsoft/BitNet-b1.58-2B-4T`:

1. [Official BitNet 2B Source Map](../bitnet/official-2b/README.md)
   - Records the official model family, current I2_S/QK256 answer authority,
     route split, claim boundaries, and source-of-truth stack.
2. [Official BitNet 2B Implementation Plan](../../plans/official-bitnet-2b/implementation-plan.md)
   - Sequences source-map, proposal, specs, CPU/CUDA excellence, Apple/A770,
     TL1/TL2, BF16/GPU-int2, and product-polish work.
3. [Official BitNet 2B Campaign](../tracking/campaigns/official-bitnet-2b/CAMPAIGN.md)
   - Summarizes the active campaign boundaries and route-specific proof model.

The current I2_S/QK256 row may remain bounded product-CLI-ready exactly as
recorded in `ci/model-artifacts/model-coverage-matrix.toml`. Do not promote
speedup, full residency, broad server readiness, TL1, TL2, BF16/GPU-int2,
Apple, or A770 claims without route-specific specs and receipts. Dense SLM
proof and diagnostic no-scale F32 QK256 proof do not satisfy production
BitNet packed I2_S/QK256 proof.

## Falcon-E Family Compact 1.58-bit Lane

Use these specs before implementing or claiming Falcon-E support:

1. [Falcon-E Family Proposal](../proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md)
   - Explains why Falcon-E is a compact direct-GGUF 1.58-bit validation lane and why it remains separate from Microsoft BitNet, 1bitLLM, Falcon3, and dense Falcon.
2. [Falcon-E Artifact Contract](BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT.md)
   - Defines the 1B and 3B direct `I2_S` GGUF candidate artifacts and inventory receipt fields.
3. [Falcon-E Route Compatibility](BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY.md)
   - Defines x86/ARM `I2_S`, `TL1`, and `TL2` initial route status and diagnostic boundaries.
4. [Falcon-E Tokenizer and Prompt](BITNET-SPEC-FALCON-E-FAMILY-TOKENIZER-PROMPT.md) and [Reference Quality](BITNET-SPEC-FALCON-E-FAMILY-REFERENCE-QUALITY.md)
   - Define tokenizer/prompt authority and bounded reference answer gates.
5. [Falcon-E I2_S](BITNET-SPEC-FALCON-E-FAMILY-I2S.md) and [TL1/TL2](BITNET-SPEC-FALCON-E-FAMILY-TL1-TL2.md)
   - Define layout-proof requirements before QK256 aliasing or TL backend work.
6. [Falcon-E CPU](BITNET-SPEC-FALCON-E-FAMILY-CPU.md), [CUDA](BITNET-SPEC-FALCON-E-FAMILY-CUDA.md), [Apple](BITNET-SPEC-FALCON-E-FAMILY-APPLE.md), [A770/OpenCL](BITNET-SPEC-FALCON-E-FAMILY-A770-OPENCL.md), and [Performance](BITNET-SPEC-FALCON-E-FAMILY-PERFORMANCE.md)
   - Define backend and benchmark proof ladders with fallback-explicit receipts.

Do not proceed from direct GGUF availability, upstream support listings, or one
Falcon-E size to answer, backend, speed, server, or full-residency claims. Claims
require exact artifact, tokenizer/prompt, layout, quality, and backend receipts
for the specific model size and route.

## Falcon3 Family Onboarding

Use these artifacts before implementing or claiming Falcon3 Family support:

1. [Falcon3 Family Proposal](../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
   - Explains why Falcon3 is the first multi-size BitNet-family onboarding lane and why 1B/7B direct GGUFs precede 3B/10B conversion routes.
2. [Falcon3 Artifact Contract](BITNET-SPEC-FALCON3-FAMILY-ARTIFACT-CONTRACT.md)
   - Defines exact artifact IDs, required inventory receipts, nominal/HF-displayed size recording, and no-binary rules.
3. [Falcon3 Route Compatibility](BITNET-SPEC-FALCON3-FAMILY-ROUTE-COMPATIBILITY.md)
   - Mirrors listed upstream I2_S/TL route support as unpromoted runner-verification candidates.
4. [Falcon3 Tokenizer and Prompt Contract](BITNET-SPEC-FALCON3-FAMILY-TOKENIZER-PROMPT.md)
   - Requires tokenizer hashes, chat-template authority, prompt token IDs, stop policy, and reference-runner command before answer claims.
5. [Falcon3 Reference Quality Contract](BITNET-SPEC-FALCON3-FAMILY-REFERENCE-QUALITY.md)
   - Defines tiny smoke, answer corpus, behavior, and long-decode gates.
6. [Falcon3 I2_S Contract](BITNET-SPEC-FALCON3-FAMILY-I2S.md)
   - Defines I2_S/QK256 layout verification before kernel aliasing.
7. [Falcon3 TL1/TL2 Contract](BITNET-SPEC-FALCON3-FAMILY-TL1-TL2.md)
   - Keeps TL routes separate from I2_S/QK256 and requires scalar TL oracles.
8. [Falcon3 CPU Contract](BITNET-SPEC-FALCON3-FAMILY-CPU.md), [CUDA Contract](BITNET-SPEC-FALCON3-FAMILY-CUDA.md), [Apple Contract](BITNET-SPEC-FALCON3-FAMILY-APPLE.md), [A770/OpenCL Contract](BITNET-SPEC-FALCON3-FAMILY-A770-OPENCL.md), and [Performance Contract](BITNET-SPEC-FALCON3-FAMILY-PERFORMANCE.md)
   - Define exact backend and benchmark gates after artifact, tokenizer/prompt, reference-good, and layout proof.
9. [Falcon3 Family Plan](../../plans/falcon3-family/implementation-plan.md)
   - Defines the PR order from registered candidates through artifact inventory, CPU proof, accelerated backends, benchmarks, product CLI, and server exact-profile work.

Do not proceed from Falcon3 registration, upstream listed support, Falcon-E
evidence, Microsoft BitNet 2B evidence, Llama3-8B-1.58 evidence, dense Falcon3
evidence, or dense SLM evidence to Falcon3 answer/backend/speed/server claims.
Claims are exact artifact, route, backend, and profile scoped.

## NPU productization

Use these artifacts before implementing or claiming NPU product support:

1. [BitNet-rs NPU Source of Truth](../npu/README.md)
   - Maps current Intel NPU evidence, not-claims, future NPU families, and
     source-of-truth artifacts.
2. [Intel Lunar Lake NPU Roadmap](intel-lunar-lake-npu-roadmap.md)
   - Defines the current Intel AI Boost NPU through OpenVINO validation lane,
     proof levels, OpenVINO constraints, and static-shape claim boundary.
3. [NPU Productization Plan](../../plans/npu/implementation-plan.md)
   - Defines the PR-by-PR sequence from governance cleanup to dense-SLM
     warm/resident route promotion and future NPU research lanes.
4. [OpenVINO Route Contract](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
   - Defines existing OpenVINO dense-SLM route labels and fallback rules that
     future NPU contracts must preserve.

Do not proceed from NPU detection, OpenVINO visibility, static graph smoke, or
static BitNet-shaped subgraph parity to full inference, QK256 decode, speedup,
full residency, or generic NPU claims. CPU fallback, OpenVINO GPU, and Arc 140V
OpenCL evidence do not count as selected NPU proof.

## AMD ROCm Productization

Use these specs before implementing or claiming AMD ROCm support:

1. [AMD ROCm Productization Proposal](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
   - Explains why ROCm exists as a selected-device AMD GPU lane and why HIP,
     BitNet QK256, dense SLM, quality, speed, residency, and server proof stay
     separate.
2. [ROCm Route Contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md)
   - Defines concrete backend labels, route IDs, proof families, receipt fields,
     and forbidden generic proof labels.
3. [ROCm Device Identity](BITNET-SPEC-ROCM-DEVICE-IDENTITY.md)
   - Defines Linux ROCm and Windows HIP SDK identity fields, official support
     status, and selected-device acceptance rules.
4. [ROCm Kernel Compile](BITNET-SPEC-ROCM-KERNEL-COMPILE.md)
   - Separates source embedding, HIP compile, runtime launch, fixture launch,
     and model route launch proof.
5. [ROCm BitNet QK256](BITNET-SPEC-ROCM-BITNET-QK256.md)
   - Defines packed I2_S/QK256 ROCm semantics, kernel IDs, fixtures, and hard
     boundaries against dense SLM and diagnostic F32 proof.
6. [ROCm Dense SLM](BITNET-SPEC-ROCM-DENSE-SLM.md)
   - Defines dense ROCm model candidates and the proof ladder separate from
     BitNet QK256.
7. [ROCm Quality](BITNET-SPEC-ROCM-QUALITY.md)
   - Defines answer-quality corpora, generated-token evidence, and failure
     taxonomy for BitNet and dense SLM ROCm.
8. [ROCm Performance](BITNET-SPEC-ROCM-PERFORMANCE.md)
   - Defines exact profiles, timing fields, comparators, and speed promotion
     gates.
9. [ROCm Residency](BITNET-SPEC-ROCM-RESIDENCY.md)
   - Defines residency classes and per-phase residency evidence.
10. [ROCm Status Surface](BITNET-SPEC-ROCM-STATUS-SURFACE.md)
    - Defines future `rocm doctor`, `model status`, `receipts explain`, and
      AMD GPU doctor status fields.

The ROCm lane is currently registered/scaffold only. Do not proceed from source
text checks, ROCm installation paths, HIP visibility, or generic AMD GPU labels
to compile, execution, model, quality, speed, residency, or server claims.

## bitnet_b1_58-large Control Model

Use these specs before implementing or claiming `1bitLLM/bitnet_b1_58-large`
artifact, conversion, tokenizer, reference, CPU, CUDA, Apple, performance, CLI,
or server support:

1. [B158 Large Artifact Contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md)
   - Defines source files, hashes, receipts, and claim boundaries.
2. [B158 Large Conversion](BITNET-SPEC-B158-LARGE-CONVERSION.md)
   - Defines safetensors inspection, F16 structural/reference GGUF, and future
     upstream-compatible `I2_S`/`TL1`/`TL2` conversion lanes.
3. [B158 Large Tokenizer and Prompt](BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md)
   - Defines tokenizer, pre-tokenizer, prompt-template, rendered prompt, and
     prompt token authority.
4. [B158 Large Reference Quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md)
   - Defines reference-runner corpus promotion and failure boundaries.
5. [B158 Large CPU](BITNET-SPEC-B158-LARGE-CPU.md)
   - Defines scalar, AVX2, AVX-512, Apple CPU/NEON, and optional Kaby CPU
     receipts after reference-good output.
6. [B158 Large CUDA](BITNET-SPEC-B158-LARGE-CUDA.md)
   - Defines CUDA route planning, one-token, short-decode, warm-session, parity,
     and no-speedup boundaries.
7. [B158 Large Apple](BITNET-SPEC-B158-LARGE-APPLE.md)
   - Defines MacBook artifact probes, M4 CPU/NEON proof, and no-Metal claim
     boundaries.
8. [B158 Large Performance](BITNET-SPEC-B158-LARGE-PERFORMANCE.md)
   - Defines exact-profile benchmark fields and speedup promotion gates.

Do not proceed from upstream support, HF safetensors presence, or F16 structural
conversion to answer, backend, or speed claims. Claims require exact artifact,
tokenizer, prompt, reference, backend, fallback, and benchmark receipts.

## BitNet b1.58 3B TL Candidate

Use these specs before implementing or claiming support for
`1bitLLM/bitnet_b1_58-3B`:

1. [3B Artifact Contract](BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md)
   - Defines exact source revision, shard, tokenizer/config, hash, storage, and
     claim-boundary requirements.
2. [3B Conversion Contract](BITNET-SPEC-B158-3B-CONVERSION.md)
   - Defines allowed safetensors, reference-runner, TL1/TL2 conversion, F16
     structural, and third-party GGUF diagnostic lanes.
3. [3B TL1/TL2 Layout](BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md)
   - Defines the route-specific TL layout fields and scalar-oracle prerequisite
     before accelerator work.
4. [3B Tokenizer and Prompt](BITNET-SPEC-B158-3B-TOKENIZER-PROMPT.md)
   - Defines tokenizer hashes, special token IDs, prompt rendering, token IDs,
     stop policy, and reference runner mode.
5. [3B Reference Quality](BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md)
   - Defines tiny smoke, answer corpus, behavior, and long-decode reference
     gates.
6. [3B CPU](BITNET-SPEC-B158-3B-CPU.md)
   - Defines x86 TL2 and ARM TL1 CPU proof paths and `I2_S` diagnostic
     rejection boundaries.
7. [3B CUDA](BITNET-SPEC-B158-3B-CUDA.md)
   - Defines the CUDA TL2 path after x86 TL2 CPU answer proof.
8. [3B Apple](BITNET-SPEC-B158-3B-APPLE.md)
   - Defines the ARM TL1 Apple CPU/NEON path and MacBook/M4/Metal boundaries.
9. [3B Performance](BITNET-SPEC-B158-3B-PERFORMANCE.md)
   - Defines exact-profile benchmark metrics and promotion rules.

Do not treat the 3B model as an `I2_S`/QK256 sibling of the official Microsoft
2B artifact. x86 TL2 and ARM TL1 remain runner/conversion candidates until
route-specific receipts prove them; all answer, backend, server, and speed
claims remain false before the shared artifact gate and these specs pass.

## Llama3 8B 1.58 Supported-Model Candidate

Use these specs before implementing or claiming support for
`HF1BitLLM/Llama3-8B-1.58-100B-tokens`:

1. [Artifact Contract](BITNET-SPEC-LLAMA3-8B-158-ARTIFACT-CONTRACT.md)
   - Defines exact revision, file hashes, tokenizer/config hashes, HF metadata,
     and the upstream/HF identity discrepancy required before artifact claims.
2. [Conversion](BITNET-SPEC-LLAMA3-8B-158-CONVERSION.md)
   - Defines safetensors, reference-runner, F16 structural, I2_S, TL1, TL2,
     and third-party GGUF lanes and their claim boundaries.
3. [Tokenizer and Prompt](BITNET-SPEC-LLAMA3-8B-158-TOKENIZER-PROMPT.md)
   - Defines the Llama3-derived tokenizer/prompt audit and forbids unproven
     Microsoft 2B prompt inheritance.
4. [Route Compatibility](BITNET-SPEC-LLAMA3-8B-158-ROUTE-COMPATIBILITY.md)
   - Records upstream-listed x86 I2_S/TL2 and ARM I2_S/TL1 routes as
     `listed_supported_verify_runner` candidates only.
5. [Reference Quality](BITNET-SPEC-LLAMA3-8B-158-REFERENCE-QUALITY.md)
   - Defines the bounded reference-good corpora and pass criteria.
6. [I2_S](BITNET-SPEC-LLAMA3-8B-158-I2S.md)
   - Defines the model-specific I2_S/QK256 layout proof and kernel IDs.
7. [TL1/TL2](BITNET-SPEC-LLAMA3-8B-158-TL1-TL2.md)
   - Defines TL layout/oracle requirements separately from QK256.
8. [CPU](BITNET-SPEC-LLAMA3-8B-158-CPU.md),
   [CUDA](BITNET-SPEC-LLAMA3-8B-158-CUDA.md),
   [Apple](BITNET-SPEC-LLAMA3-8B-158-APPLE.md), and
   [Performance](BITNET-SPEC-LLAMA3-8B-158-PERFORMANCE.md)
   - Define backend and benchmark gates after artifact, tokenizer, conversion,
     reference-good, and layout proofs pass.

Do not treat this model as “the official Microsoft 2B model but bigger.” The
current allowed claim is registered candidate only: upstream route support is
known, the HF safetensors artifact is visible, and BitNet-rs has conservative
onboarding contracts.

## NPU Productization

Use these docs before implementing or claiming NPU support:

1. [NPU Source-of-Truth Map](../npu/README.md)
   - Maps current Intel NPU evidence, claim boundaries, source-of-truth layers,
     and future NPU families.
2. [NPU Productization Proposal](../proposals/BITNET-PROP-0007-npu-productization.md)
   - Defines why NPUs are the low-power / resident inference lane rather than a
     generic accelerator bucket.
3. [Intel Lunar Lake NPU Roadmap](intel-lunar-lake-npu-roadmap.md)
   - Defines the existing Intel NPU proof levels, OpenVINO constraints, and
     NPU-002 through NPU-011 evidence boundary.
4. [NPU Route Contract](BITNET-SPEC-NPU-ROUTE-CONTRACT.md)
   - Defines route IDs, backend labels, receipt fields, proof-family separation,
     and non-conflation rules.
5. [NPU Proof Ladder](BITNET-SPEC-NPU-PROOF-LADDER.md)
   - Defines maturity levels from detection through complete full-route proof.
6. [NPU Cold/Warm/Cache Contract](BITNET-SPEC-NPU-COLD-WARM-CACHE.md)
   - Defines cold, cached, warm, and resident timing fields and promotion rules.
7. [NPU Dense SLM Contract](BITNET-SPEC-NPU-DENSE-SLM.md)
   - Defines Qwen2.5/Qwen3-class OpenVINO GenAI dense SLM candidate routes.
8. [NPU BitNet Subgraph Contract](BITNET-SPEC-NPU-BITNET-SUBGRAPH.md)
   - Defines static BitNet-shaped subgraph parity without full inference claims.
9. [NPU Quality Contract](BITNET-SPEC-NPU-QUALITY.md),
   [Performance Contract](BITNET-SPEC-NPU-PERFORMANCE.md),
   [Residency Contract](BITNET-SPEC-NPU-RESIDENCY.md), and
   [Status Surface Contract](BITNET-SPEC-NPU-STATUS-SURFACE.md)
   - Define answer quality, benchmark, residency, and user-visible status gates.
10. [NPU Implementation Plan](../../plans/npu/implementation-plan.md)
   - Defines PR-sized sequencing and proof commands for the NPU productization
     campaign.

Do not proceed from NPU detection, OpenVINO NPU smoke, or static BitNet-shaped
subgraph parity to full inference, speedup, packed QK256, generic NPU, or full
residency claims. NPU promotion is per model + route + profile and requires
strict fallback=false receipts.


## CPU AVX-512 Kernel Proof

Use these specs before implementing or claiming AVX-512 CPU kernel support:

1. [CPU AVX-512 Kernel Contract](BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md)
   - Defines the difference between AVX-512 detection, dispatch, execution,
     parity, performance, and sustained-performance proof.
2. [CPU ISA Selection](BITNET-SPEC-CPU-ISA-SELECTION.md)
   - Defines strict `auto`, `scalar`, `avx2`, `avx512`, and `avx512-vnni`
     request behavior and fallback receipts.
3. [AMD Ryzen 9 9950X3D CPU Roadmap](amd-9950x3d-cpu-roadmap.md)
   - Defines the 9950X3D CPU-only lane, required profiles, comparisons, and
     sustained/cache-domain metadata.
4. [CPU AVX-512 Implementation Plan](../../plans/cpu-avx512/implementation-plan.md)
   - Defines the PR-by-PR sequence from documentation rails to profile-scoped
     auto-selection promotion.

Do not proceed from AVX-512 detection or an AVX-512 receipt label to speed or
execution claims. Claims require distinct selected kernel IDs, strict fallback
truth, invocation counters, parity, and profile/sustained receipts.

## A770 BitNet Productization

Use these artifacts before processing the A770 diagnostic branch chain or
closing/replacing A770 diagnostic PRs:

1. [A770 Diagnostic Lineage](BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE.md)
   - Defines durable versus transient diagnostic content, lineage frontiers,
     successor rules, forbidden batch actions, and the no-claim boundary for
     diagnostic evidence.
2. [A770 Diagnostic Lineage Plan](../../plans/a770-diagnostic-lineage/implementation-plan.md)
   - Sequences lineage policy/checker work before narrow runtime salvage slices.

Do not bulk-close, bulk-reopen, bulk-recreate, or treat diagnostic-only as
disposable. Diagnostic lineage can route runtime salvage, but it does not prove
A770 support, quality, speed, residency, or selected execution.

Use these specs before implementing or claiming A770 BitNet product support:

1. [A770 BitNet Claim Boundary](a770-bitnet-claim-boundary.md)
   - Defines claim levels, required evidence, current not-claims, and the
     selected-attention boundary for real BitNet question-answer usage on
     A770.
2. [Intel Arc A770 GPU Roadmap](intel-arc-a770-gpu-roadmap.md)
   - Defines the A770 hardware/runtime lane and native OpenCL proof path.
3. [A770 BitNet Productization Plan](../../plans/a770-bitnet-claim-boundary-implementation.md)
   - Defines the PR-by-PR implementation sequence and acceptance gates.

Do not proceed from A770 detection or a single benchmark to product claims.
Claims must pass through the claim boundary and productization plan first.

# BitNet.cpp API Discovery - Documentation Index

**Discovery Date**: October 25, 2025
**Status**: COMPLETE - All APIs identified and documented

## A770 BitNet Productization

Use these specs before implementing or claiming A770 BitNet product support:

1. [A770 BitNet Claim Boundary](a770-bitnet-claim-boundary.md)
   - Defines claim levels, required evidence, current not-claims, and the
     selected-attention boundary for real BitNet question-answer usage on
     A770.
2. [Intel Arc A770 GPU Roadmap](intel-arc-a770-gpu-roadmap.md)
   - Defines the A770 hardware/runtime lane and native OpenCL proof path.
3. [A770 BitNet Productization Plan](../../plans/a770-bitnet-claim-boundary-implementation.md)
   - Defines the PR-by-PR implementation sequence and acceptance gates.

Do not proceed from A770 detection or a single benchmark to product claims.
Claims must pass through the claim boundary and productization plan first.

## Overview

This directory contains comprehensive documentation of the BitNet.cpp API discovered during G2 discovery phase. BitNet.cpp uses the llama.cpp C API; there is no separate BitNet-specific public interface.

## Files in This Directory

### 0. apple-silicon-backend-roadmap.md (New)
**Purpose**: Phased implementation plan for Apple GPU/NPU support in BitNet-rs
**Audience**: Runtime/kernel/inference contributors planning Apple backend work

**Contents**:
- Current-state backend audit
- Device abstraction and probing changes
- Metal-first backend plan
- Optional Core ML/MLCompute (ANE) integration path
- Feature-flag strategy, roadmap, risks, and success criteria

**When to use**: Planning and sequencing Apple Silicon acceleration work

### 1. bitnet-cpp-api-requirements.md (Full Reference)
**Purpose**: Complete API documentation with implementation examples
**Audience**: Developers implementing the wrapper
**Length**: 371 lines

**Contents**:
- Executive summary
- Available artifacts location
- Complete API reference (6 sections)
- Model loading, context creation, tokenization, batching, decoding, logits
- Implementation patterns for both wrapper functions
- Build configuration and CMake integration
- API differences from commented code
- Potential issues and workarounds
- Validation checklist
- Recommended implementation order

**When to use**: For detailed implementation, full API reference, build setup

### 2. bitnet-cpp-api-quick-reference.md (Cheat Sheet)
**Purpose**: Quick lookup of functions and patterns
**Audience**: All developers
**Length**: 161 lines

**Contents**:
- One-liner function table with line numbers
- Critical model and context parameters
- Minimal working examples (tokenization, inference)
- Error handling pattern
- Required includes
- Memory management summary
- Logits layout and indexing
- Two-pass API patterns
- Build configuration quick reference
- What's NOT needed (no BitNet-specific API)

**When to use**: Quick lookup, code patterns, parameter reference

### 3. bitnet-cpp-wrapper-implementation-guide.md (Step-by-Step)
**Purpose**: Implementation roadmap with specific steps
**Audience**: Developer implementing `bitnet_cpp_wrapper.cc`
**Length**: 241 lines

**Contents**:
- File structure and current state
- TODO-to-API mapping for both functions
- Line-by-line implementation steps for tokenization (lines 87-157)
- Line-by-line implementation steps for evaluation (lines 224-312)
- Critical changes from commented code
- Feature flag configuration
- Recommended implementation phases (tokenization, then inference, then integration)
- Testing strategy (unit, integration, acceptance)
- Environment setup
- Build configuration details
- Debugging tips
- Performance expectations
- Known limitations and future improvements
- Success metrics

**When to use**: Planning implementation, step-by-step guidance, testing

## Quick Start Paths

### "I need to implement the wrapper now"
1. Read: Quick Reference (2 min)
2. Read: Implementation Guide Phase 1 (10 min)
3. Start coding tokenization
4. Reference: Full API Requirements as needed

### "I need to understand the API"
1. Read: Full API Requirements (20 min)
2. Review: Implementation examples (5 min)
3. Check: Quick Reference for specific functions

### "I'm debugging a specific function"
1. Search: Quick Reference for function
2. Look up: Full requirements for detailed docs
3. Check: Implementation guide for expected behavior

## Key Findings Summary

### Location
- **Root**: `/home/steven/.cache/bitnet_cpp/`
- **Headers**: `3rdparty/llama.cpp/include/llama.h`
- **Libraries**: `build/lib/libllama.so`, `libggml.so`

### Core API Functions Needed

| Task | Function | Status |
|------|----------|--------|
| Load model | `llama_load_model_from_file()` | ✓ Available |
| Create context | `llama_new_context_with_model()` | ✓ Available |
| Tokenize | `llama_tokenize()` | ✓ Available |
| Decode | `llama_decode()` | ✓ Available |
| Get logits | `llama_get_logits()` | ✓ Available |
| Get vocab size | `llama_n_vocab()` | ✓ Available |
| Create batch | `llama_batch_get_one()` | ✓ Available |
| Free resources | `llama_free()`, `llama_free_model()` | ✓ Available |

### Critical Parameters

**For Model Loading**:
```c
use_mmap = true      // Enable memory mapping
n_gpu_layers = 0     // CPU only (for MVP)
```

**For Context**:
```c
n_ctx = 2048         // Context size
logits_all = true    // CRITICAL: all-position logits
n_threads = 4        // CPU threads
```

### Two-Pass API Pattern
Both wrapper functions use two-pass buffer negotiation:
1. **Pass 1**: NULL buffer query returns size
2. **Pass 2**: Real buffer call copies data

### Implementation Estimate
- **Phase 1 (Tokenization)**: 1-2 hours
- **Phase 2 (Inference)**: 2-3 hours
- **Phase 3 (Integration)**: 1-2 hours
- **Total**: 4-7 hours

## Important Notes

### What's NOT Needed
- ❌ `bitnet_get_tokenizer()` - doesn't exist
- ❌ `bitnet_tokenize_text()` - doesn't exist
- ❌ `bitnet_eval_all_positions()` - doesn't exist

Use `llama_*` functions directly instead.

### What's Different from Comments
The uncommented code in `bitnet_cpp_wrapper.cc` mentions hypothetical functions. The actual API is:
- Tokenization: Use `llama_tokenize()` directly on model
- Inference: Use `logits_all=true` parameter, not a separate function
- All else: Matches the comments closely

### Performance Notes
- Current MVP loads model per-call (inefficient)
- Will be optimized in v0.2 with context caching
- Expected: ~0.1-1 tok/s for 2B models on CPU
- Per-call overhead: ~100-500ms (model load)

## Comprehensive Documentation Suite

### Complete BitNet.cpp Integration Guides

1. **[BitNet.cpp API Requirements](bitnet-cpp-api-requirements.md)** (Full Reference)
   - Complete API documentation with implementation examples
   - All llama.cpp functions needed for wrapper
   - Build configuration and setup

2. **[BitNet.cpp API Quick Reference](bitnet-cpp-api-quick-reference.md)** (Cheat Sheet)
   - Quick lookup of functions and patterns
   - Minimal working examples
   - Common parameters at a glance

3. **[BitNet.cpp Wrapper Implementation Guide](bitnet-cpp-wrapper-implementation-guide.md)** (Step-by-Step)
   - Implementation roadmap with specific steps
   - Line-by-line guidance for both FFI functions
   - Testing strategy and success criteria

4. **[BitNet.cpp AVAILABLE Mode Wiring Guide](bitnet-available-wiring.md)** ⭐
   - **Comprehensive wiring guide for production FFI integration**
   - Required headers and library dependencies
   - Build system configuration (build.rs)
   - Symbol visibility and linking best practices
   - Platform-specific notes (Linux, macOS, Windows)
   - Common compilation errors with fixes
   - Troubleshooting guide with diagnostics
   - Verification checklist

5. **[BitNet.cpp FFI Integration Sockets](bitnet-cpp-ffi-sockets.md)** 🆕 LATEST
   - **Technical specification for 6 missing FFI sockets**
   - Context initialization for persistent model loading
   - BitNet-specific tokenization and inference (optional, with llama.cpp fallback)
   - Session API for 10-100× performance improvements
   - dlopen loader architecture for runtime symbol resolution
   - Graceful degradation when symbols unavailable
   - Migration path from current per-call model loading to session API
   - Testing strategy and performance benchmarks

6. **[BitNet.cpp Session API](bitnet-session-api.md)** 📘
   - High-level session management design
   - Lifecycle management and resource cleanup
   - Integration with FFI sockets
   - Performance optimization strategies

### When to Use Which Guide

| Task | Recommended Guide |
|------|-------------------|
| Understanding the API | API Requirements (Full Reference) |
| Quick function lookup | API Quick Reference (Cheat Sheet) |
| Implementing wrapper code | Wrapper Implementation Guide |
| Build system integration | AVAILABLE Mode Wiring Guide ⭐ |
| Troubleshooting linker errors | AVAILABLE Mode Wiring Guide ⭐ |
| Platform-specific issues | AVAILABLE Mode Wiring Guide ⭐ |
| Symbol visibility problems | AVAILABLE Mode Wiring Guide ⭐ |
| Designing persistent context API | FFI Integration Sockets 🆕 |
| Runtime symbol resolution | FFI Integration Sockets 🆕 |
| Session management architecture | Session API + FFI Integration Sockets |
| Performance optimization (10-100×) | FFI Integration Sockets 🆕 |
| dlopen loader implementation | FFI Integration Sockets 🆕 |

## Cross-Validation Specifications

### Parity-Both Command Suite

7. **[Parity-Both Command](parity-both-command.md)** 🔄
   - **Baseline dual-lane cross-validation specification**
   - Single command orchestration for BitNet.cpp + llama.cpp
   - Unified receipt generation and comparative metrics
   - Exit code semantics (0=both pass, 1=either fails)
   - Command-line interface and workflow architecture

8. **[Parity-Both Preflight & TokenizerAuthority Integration](parity-both-preflight-tokenizer-integration.md)** 🆕 COMPREHENSIVE
   - **Complete integration specification for v2.0**
   - Preflight integration with auto-repair and retry semantics
   - TokenizerAuthority metadata for receipt reproducibility
   - Exit code standardization (0=both pass, 1=one fails, 2=usage error)
   - Dual receipt generation with shared tokenizer provenance
   - Token parity validation as fail-fast gate
   - Per-position metrics with configurable thresholds
   - 10 acceptance criteria (AC1-AC10) with detailed implementation guidance
   - 54 comprehensive tests covering all acceptance criteria
   - API contracts and function signatures
   - Receipt schema v2.0.0 (backward-compatible extension)

9. **[Preflight Auto-Repair](preflight-auto-repair.md)** 🔧
   - Auto-repair integration for missing C++ backends
   - RepairMode semantics (Auto, Never, Always)
   - Retry logic with exponential backoff
   - setup-cpp-auto integration

10. **[Preflight Repair Mode Re-exec](preflight-repair-mode-reexec.md)** 🔄
    - Re-execution semantics after backend repair
    - Environment variable propagation
    - Build system integration with xtask rebuild

### When to Use Which Parity-Both Guide

| Task | Recommended Guide |
|------|-------------------|
| Understanding baseline dual-lane flow | Parity-Both Command (baseline) |
| Complete integration implementation | Parity-Both Preflight & TokenizerAuthority Integration 🆕 |
| Auto-repair mechanics | Preflight Auto-Repair |
| Re-exec after repair | Preflight Repair Mode Re-exec |
| TokenizerAuthority schema | Parity-Both Preflight & TokenizerAuthority Integration 🆕 |
| Exit code logic (0/1/2) | Parity-Both Preflight & TokenizerAuthority Integration 🆕 |
| Receipt schema v2.0.0 | Parity-Both Preflight & TokenizerAuthority Integration 🆕 |
| Test coverage planning (54 tests) | Parity-Both Preflight & TokenizerAuthority Integration 🆕 |

## Related Documentation

See also in parent `/docs` directory:
- `BITNET_CPP_INTEGRATION_ANALYSIS.md` - Previous analysis
- `C_FFI_INTEGRATION_ANALYSIS.md` - FFI strategy
- `CROSSVAL.md` - Cross-validation framework
- `explanation/dual-backend-crossval.md` - Dual-backend architecture

## Files to Modify

1. **`crossval/src/bitnet_cpp_wrapper.cc`**
   - Tokenization: Replace lines 87-157
   - Inference: Replace lines 224-312
   - Feature flag: `BITNET_AVAILABLE` vs `BITNET_STUB`

2. **`crossval/build.rs`** (minor)
   - Ensure llama.so, ggml.so linked
   - Set include path for llama.h

3. **`Cargo.toml`** (maybe)
   - Add feature flag definition if needed

## Success Criteria

All items should be checked before implementation is complete:

- [ ] Both functions compile with `BITNET_AVAILABLE`
- [ ] Tokenization output matches llama.cpp baseline
- [ ] Logits shape is `[n_tokens][n_vocab]`
- [ ] All-positions logits in single decode call
- [ ] Error handling for all failure paths
- [ ] Memory properly freed in all paths
- [ ] No memory leaks (valgrind clean)
- [ ] Integration tests pass
- [ ] Cross-validation tests pass
- [ ] Performance profiled and documented

## Implementation Workflow

```
1. Create feature-branch
   git checkout -b feat/bitnet-cpp-available-mode

2. Phase 1: Tokenization
   - Implement crossval_bitnet_tokenize()
   - Test tokenization end-to-end
   - Commit

3. Phase 2: Inference
   - Implement crossval_bitnet_eval_with_tokens()
   - Test inference end-to-end
   - Commit

4. Phase 3: Integration
   - Run cross-validation tests
   - Profile performance
   - Document results
   - Create pull request

5. Review & Merge
   - Code review
   - CI passes
   - Merge to main
```

## Getting Help

1. **Quick question?** → Check Quick Reference
2. **Implementation stuck?** → Check Implementation Guide
3. **API details?** → Check Full Requirements
4. **Specific error?** → Debug Tips section in Implementation Guide

## Document Versions

- v1.0 (Oct 25, 2025): Initial discovery, all APIs documented
- (Future versions as implementation progresses)

---

**Last Updated**: October 25, 2025
**Next Update**: After Phase 1 implementation complete

## C++ Wrapper KV Position Tracking

**File**: `cpp-wrapper-kv-position-tracking.md`
**Version**: 1.0.0
**Date**: 2025-10-25
**Status**: Ready for Implementation (v0.2)

### Overview

Technical specification for implementing manual KV cache position tracking in the C++ wrapper to replace the removed `llama_get_kv_cache_token_count()` API. Enables multi-turn conversation support and autoregressive generation with 10-100× performance improvement.

### Key Features

- Manual `n_past` position tracking in `bitnet_context_t`
- Position validation to prevent KV cache overflow
- Context reset API for new conversations
- Migration path from Socket 0 (stateless) to Socket 1 (persistent)
- Backward compatibility with existing stateless evaluation

### Target Release

- v0.2: Core position tracking implementation
- v0.3: Advanced features (sliding window, multi-sequence batching)

### Related Documents

- `bitnet-cpp-ffi-sockets.md` - Socket 1/Socket 3 architecture
- `dual-backend-crossval.md` - Cross-validation patterns
- `cpp-setup.md` - C++ reference setup guide

## Qwen3.6 Modern Dense/Multimodal Family

Use these artifacts before implementing or claiming Qwen3.6 support:

1. [Qwen3.6 Source Map](../qwen/qwen36/README.md)
2. [Qwen3.6 Implementation Plan](../../plans/qwen36/implementation-plan.md)
3. [Qwen3.6 Proposal](../proposals/BITNET-PROP-0017-qwen36-modern-dense-model-family.md)
4. [Artifact Contract](BITNET-SPEC-QWEN36-FAMILY-ARTIFACT-CONTRACT.md)
5. [Processor/Tokenizer/Prompt](BITNET-SPEC-QWEN36-PROCESSOR-TOKENIZER-PROMPT.md)
6. [Architecture Inventory](BITNET-SPEC-QWEN36-ARCHITECTURE-INVENTORY.md)
7. [External Reference](BITNET-SPEC-QWEN36-EXTERNAL-REFERENCE.md)
8. [Text-only Native](BITNET-SPEC-QWEN36-TEXT-ONLY-NATIVE.md)
9. [MoE 35B-A3B](BITNET-SPEC-QWEN36-MOE-35B-A3B.md)
10. [Multimodal](BITNET-SPEC-QWEN36-MULTIMODAL.md), [Memory Envelope](BITNET-SPEC-QWEN36-MEMORY-ENVELOPE.md), [Quality](BITNET-SPEC-QWEN36-QUALITY.md), [Performance](BITNET-SPEC-QWEN36-PERFORMANCE.md), [Server](BITNET-SPEC-QWEN36-SERVER.md), [Status Surface](BITNET-SPEC-QWEN36-STATUS-SURFACE.md)

Qwen3.6 is a separate governed family lane: not BitNet proof, not inherited Qwen2.5/Qwen3 0.6B proof, and external sidecar/API evidence is not native Rust proof.

## TL2 Productization

Use these artifacts before implementing or claiming TL2 route support:

1. [TL2 Source Map](../bitnet/tl2/README.md)
2. [TL2 Implementation Plan](../../plans/tl2/implementation-plan.md)
3. [TL2 Productization Proposal](../proposals/BITNET-PROP-0018-tl2-productization.md)
4. [TL2 Route Contract](BITNET-SPEC-TL2-ROUTE-CONTRACT.md)
5. [TL2 Layout](BITNET-SPEC-TL2-LAYOUT.md)
6. [TL2 Scalar Oracle](BITNET-SPEC-TL2-SCALAR-ORACLE.md)
7. [TL2 X86 AVX](BITNET-SPEC-TL2-X86-AVX.md)
8. [TL2 Artifact Gate](BITNET-SPEC-TL2-ARTIFACT-GATE.md)
9. [TL2 Model Compatibility](BITNET-SPEC-TL2-MODEL-COMPATIBILITY.md)
10. [TL2 Reference Quality](BITNET-SPEC-TL2-REFERENCE-QUALITY.md)
11. [TL2 CPU](BITNET-SPEC-TL2-CPU.md), [CUDA](BITNET-SPEC-TL2-CUDA.md), [Performance](BITNET-SPEC-TL2-PERFORMANCE.md), and [Status Surface](BITNET-SPEC-TL2-STATUS-SURFACE.md)

TL2 is x86-first and separate from I2_S/QK256 and TL1 proof families.
