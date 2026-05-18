# BITNET-SPEC-ROCM-DEVICE-IDENTITY: AMD ROCm Device Identity

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines identity fields; no runtime promotion
Policy impact: no CI policy exception

## Purpose

Make AMD GPU identity precise enough that ROCm receipts can distinguish an
installed toolkit, a visible HIP runtime, an officially supported selected GPU,
an unsupported but visible GPU, and a model route that actually executed.

## Required Linux Fields

Linux ROCm receipts must record:

```text
OS/distro/kernel/glibc
ROCm version
HIP version
rocminfo available
rocm-smi available
hipconfig available
ROCM_PATH / HIP_PATH
GPU name
GFX target
architecture family
PCI bus ID
VRAM
compute units
wavefront size
max workgroup size
driver version
amdgpu kernel driver
render/video device permissions
multi-GPU topology
XGMI/PCIe link info where available
```

## Required Windows Fields

Windows HIP SDK receipts must record:

```text
Windows version
HIP SDK version
AMD driver version
GPU name
GFX target
HIP runtime availability
HIP SDK availability
debugger availability if relevant
```

## Official Support Status

A receipt must distinguish these states:

```text
supported_official
unsupported_official
community_enabled
unknown
```

Official support status is versioned evidence. It must be recorded with the
ROCm or HIP SDK documentation/runtime version consulted at proof time and must
not silently carry forward to later versions.

## Acceptance Rules

- Linux ROCm proof and Windows HIP SDK proof are separate.
- A receipt must not say "ROCm available" if it only found `/opt/rocm` but no
  GPU.
- A receipt must not say "selected AMD GPU" unless runtime/device identity is
  resolved.
- A selected backend must be concrete, such as
  `amd-radeon-rx-7900-xtx-rocm`, not `amd`, `gpu`, `rocm`, `hip`, or `radeon`.
- Unsupported official status does not block diagnostic receipts, but it blocks
  user-facing support promotion unless a policy exception explicitly allows the
  exact scope.
