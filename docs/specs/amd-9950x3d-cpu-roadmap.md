# AMD Ryzen 9 9950X3D CPU Roadmap

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`
- `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
Linked ADRs: n/a
Linked plan:
- `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: 9950X3D AVX-512 claims remain receipt-scoped and CPU-only.
Policy impact: No policy exception.

## Purpose

This lane validates the modern high-end AMD desktop CPU path for BitNet-rs.

Primary proof label:

```text
amd-9950x3d-cpu-avx512
```

Secondary comparison labels:

```text
amd-9950x3d-cpu-avx2
amd-9950x3d-cpu-scalar
```

The 9950X3D lane is CPU-only. It is not a GPU or NPU acceleration lane.

## Hardware Baseline

| Property | Expected value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D |
| Architecture | Zen 5 / Granite Ridge |
| Socket | AM5 |
| Cores / threads | 16 / 32 |
| Base / boost | 4.3 GHz / up to 5.7 GHz |
| L3 cache | 128 MB |
| TDP | 170 W |
| Memory | DDR5 |
| PCIe | PCIe 5.0 |
| Extensions | AVX-512, AVX2, AVX, FMA3 |
| Cooling | Liquid cooler recommended for optimal performance |

This is a dual-CCD X3D CPU. Receipts should record scheduler, core placement, and cache-domain context when available. Do not assume one timing number describes the whole processor.

## Claim Boundary

- AVX-512 detection is not AVX-512 kernel proof.
- AVX2 proof is not AVX-512 proof.
- Short boost behavior is not sustained performance.
- X3D/cache-sensitive wins must be tied to benchmark receipts.
- CPU proof is not GPU/NPU proof.
- GPU/NPU fallback cannot be involved in strict CPU proof.

## Validation Levels

| Level | Evidence | Allowed claim |
|---|---|---|
| 0 | CPU model detected | 9950X3D detected |
| 1 | Runtime feature probe records AVX2 and AVX-512 | CPU feature profile recorded |
| 2 | Scalar, AVX2, and AVX-512 kernel smoke pass | CPU kernel smoke tested |
| 3 | Strict CPU inference receipt validates | CPU proof receipt backed |
| 4 | Cache-sensitive and sustained-power baselines exist | Modern desktop CPU benchmark recorded |

## Required AVX-512 Profiles

The 9950X3D AVX-512 lane must record profile-specific evidence before any
profile-specific speed claim:

```text
micro_qk256_f32_gemv
micro_qk256_i8s_scaled_gemv
layer_0_decode
prefill_128
prefill_512
first_token
decode_32
decode_128
warm_session_3_turns
sustained_decode_10min
```

Required comparisons are:

```text
scalar vs avx2
scalar vs avx512
avx2 vs avx512
avx512 vs cuda diagnostic
```

The CUDA comparison is diagnostic only. It does not turn the 9950X3D lane into
a GPU proof lane and must not be used for a CUDA readiness claim.

## Receipt Fields

Minimum CPU proof receipt:

```json
{
  "machine_id": "amd-9950x3d",
  "requested_backend": "cpu",
  "selected_backend": "amd-9950x3d-cpu-avx512",
  "fallback_backend": null,
  "fallback_used": false,
  "cpu": {
    "vendor": "AMD",
    "model": "Ryzen 9 9950X3D",
    "architecture": "Zen 5",
    "cores": 16,
    "threads": 32,
    "l3_cache_bytes": 134217728,
    "avx2_detected": true,
    "avx512_detected": true,
    "tdp_watts": 170
  },
  "cpu_topology": {
    "ccd_count": 2,
    "x3d_cache_domain": "...",
    "core_affinity": "...",
    "scheduler_policy": "...",
    "smt_enabled": true
  },
  "power": {
    "mode": "...",
    "sustained_run": true,
    "duration_seconds": 600
  }
}
```

## Work Plan

### AMD9950X3D-001 - Add CPU Lane Docs

Docs/tracking only. Add backend status, roadmap, and hardware validation profile.

### AMD9950X3D-002 - Machine Profile

Collect OS, CPU flags, topology, scheduler/core placement context, memory, governor/power mode, thermal state, and optional OpenVINO CPU visibility.

### AMD9950X3D-003 - Scalar, AVX2, and AVX-512 Dispatch Proof

Prove scalar, AVX2, and AVX-512 paths can be forced independently and receipts record requested kernel, selected kernel, CPU features, fallback status, fallback reason, and AVX-512 invocation counters. AVX-512 detection or a receipt label is not enough.

### AMD9950X3D-004 - Strict CPU Proof Run

Run strict CPU proof with no GPU/NPU fallback, no mock path, and no hidden loader fallback.

### AMD9950X3D-005 - Cache-Sensitive Benchmark Baseline

Record cache-domain, scheduler/core placement, memory, and selected CPU path context.

### AMD9950X3D-006 - Sustained-Power Benchmark Receipt

Record sustained frequency, temperature if available, power mode, cooling context, and duration.

## Relationship To Other CPU Lanes

| Machine | Role |
|---|---|
| i5-8250U | Low-power Intel AVX2 mobile baseline |
| Ryzen 7 5700X | Mainstream AMD AVX2 desktop baseline |
| Ryzen 9 9950X3D | Modern AMD AVX-512 and large-cache desktop baseline |
| M4 Mac mini | ARM64/NEON and Metal ecosystem comparison |

The 9950X3D answers:

```text
How does the CPU-first path behave on a modern high-end AVX-512 and large-cache AMD desktop?
```

## Related AVX-512 Contracts

- `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`
- `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
- `plans/cpu-avx512/implementation-plan.md`

## Do Not

- Do not treat AVX2 proof as AVX-512 proof.
- Do not report short boost as sustained performance.
- Do not ignore cache-domain or scheduler context for X3D behavior.
- Do not treat CPU proof as GPU/NPU proof.
- Do not make performance claims without sustained-power receipts.
