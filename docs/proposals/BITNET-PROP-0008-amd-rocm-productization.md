# BITNET-PROP-0008: AMD ROCm Productization

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: n/a
Linked specs: [ROCm route contract](../specs/BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm device identity](../specs/BITNET-SPEC-ROCM-DEVICE-IDENTITY.md), [ROCm kernel compile](../specs/BITNET-SPEC-ROCM-KERNEL-COMPILE.md), [ROCm BitNet QK256](../specs/BITNET-SPEC-ROCM-BITNET-QK256.md), [ROCm dense SLM](../specs/BITNET-SPEC-ROCM-DENSE-SLM.md), [ROCm quality](../specs/BITNET-SPEC-ROCM-QUALITY.md), [ROCm performance](../specs/BITNET-SPEC-ROCM-PERFORMANCE.md), [ROCm residency](../specs/BITNET-SPEC-ROCM-RESIDENCY.md), [ROCm status surface](../specs/BITNET-SPEC-ROCM-STATUS-SURFACE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: proposes an AMD ROCm lane; does not promote runtime support
Policy impact: live ROCm proof remains opt-in and outside ordinary PR CI

## Thesis

AMD ROCm support gives BitNet-rs a vendor-diverse native GPU path. The lane
should start with strict HIP/ROCm selected-device proof, then BitNet QK256 and
dense SLM route proof, then profile-specific performance and residency. It is
not generic AMD GPU support and it does not inherit CUDA, OpenCL, WGPU, CPU,
OpenVINO, or generic GPU proof.

## Why ROCm Exists As A Lane

BitNet-rs already has hardware-specific lanes for CPU, Intel OpenCL/OpenVINO,
Apple Metal, and NVIDIA CUDA. AMD GPU users need the same level of proof: exact
hardware identity, runtime identity, selected backend identity, fallback truth,
quality gates, parity, profile timing, residency, and user-visible status.

ROCm should therefore be a first-class lane for selected AMD Radeon, Radeon PRO,
and Instinct devices. The first product target must be one exact route that is
available to the project, such as `amd-radeon-rx-7900-xtx-rocm`,
`amd-radeon-pro-w7900-rocm`, or `amd-instinct-mi300x-rocm`, not an "all AMD
GPUs" promise.

## Why HIP First

HIP is the correct first runtime API because it is AMD's C++ runtime and kernel
programming interface for GPU device management, streams, memory, module
loading, runtime compilation, launch APIs, and CUDA-adjacent kernel structure.
BitNet-rs already has HIP-looking embedded kernel sources, so the next proof
step should be HIP compile and launch evidence rather than a parallel OpenCL,
WGPU, OpenVINO, or CPU path.

HIP does not make ROCm proof interchangeable with CUDA proof. A HIP kernel that
compiles for `gfx1100` proves a different runtime, compiler, device ISA, and
receipt family from a CUDA kernel on an RTX 5070 Ti.

## Why Dense SLM And BitNet QK256 Are Separate

BitNet QK256 proof requires official Microsoft I2_S/QK256 GGUF semantics,
canonical packed layout, BitNet.cpp-aligned I2_S code mapping, I8_S activation
quantization, scale and `act_sum` math, tail behavior, row stride behavior, and
strict tokenizer/template authority.

Dense SLM proof uses regular dense GGUF tensors and model-specific tokenizer,
prompt, quality, and warm-session rules. A dense Qwen ROCm route can prove dense
SLM ROCm for that artifact and profile, but it cannot prove packed BitNet
I2_S/QK256. A BitNet QK256 receipt can prove the official BitNet route, but it
cannot prove dense Qwen, Qwen3, SmolLM2, Llama, Gemma, or Phi routes.

## Why Source Text Tests Are Insufficient

Source-text tests can show that embedded `.hip` files contain expected symbols,
`__global__` markers, HIP indexing, and structural safety checks. They do not
prove that HIP headers are available, the source compiles under a specific HIP
or ROCm version, the compiler targets a specific GFX architecture, a module can
load, a kernel can launch, memory transfers work, or model inference uses the
kernel.

The ROCm ladder therefore separates source embedding, static syntax, hostless
hipcc compile, target GFX compile, HIPRTC compile, tiny kernel launch, fixture
kernel launch, and model route launch.

## Exact Device, GFX Target, And Runtime Version

ROCm support shifts across devices, operating systems, ROCm releases, and HIP
API versions. Receipts must record ROCm version, HIP version, selected AMD GPU,
architecture family, GFX target, PCI identity, VRAM, compute units, wavefront
size, driver/kernel context, and official support status where available.

A receipt must not say "ROCm available" if it only found `/opt/rocm`. A receipt
must not say "selected AMD GPU" unless the runtime/device identity is resolved.
A kernel compiled under one HIP/ROCm release does not prove another release.

## Linux And Windows Evidence Are Separate

Linux ROCm and Windows HIP SDK support have different installation, driver,
device, and tooling boundaries. Linux proof should record distro, kernel,
glibc, amdgpu driver, `rocminfo`, `rocm-smi`, `hipconfig`, `ROCM_PATH`,
`HIP_PATH`, render/video device permissions, and topology. Windows HIP SDK proof
should record Windows version, HIP SDK version, AMD driver version, HIP runtime
availability, HIP SDK availability, GPU name, GFX target, and debugger
availability when relevant.

A Windows HIP SDK receipt does not prove Linux ROCm support, and a Linux ROCm
receipt does not prove Windows HIP SDK support.

## Exact-Profile Speed And Residency

Speed and residency are profile-specific claims. BitNet-rs may record timings
as benchmark candidates before it accepts a speedup. Promotion requires quality
passed, fallback false, applicable profile timing, same-model CPU comparator,
relevant same-model CUDA/A770 comparator where useful, repeated same-device
history receipts, and review acceptance.

Full ROCm residency is also separate from partial acceleration. QK256 linears
on ROCm are not full decode residency. Dense linears on ROCm are not full model
residency. Receipts must name per-phase residency before any full-residency
claim is allowed.

## Claim Boundary

This proposal permits only a docs/status registration claim until later proof
lands. It does not promote runtime detection, selected AMD GPU support, compile
proof, execution proof, BitNet QK256 proof, dense SLM proof, answer quality,
speedup, full residency, product CLI readiness, or server readiness.
