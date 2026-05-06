# AMD Ryzen 9 9950X3D CPU Roadmap

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
  "power": {
    "mode": "...",
    "sustained_run": true
  }
}
```

## Work Plan

### AMD9950X3D-001 - Add CPU Lane Docs

Docs/tracking only. Add backend status, roadmap, and hardware validation profile.

### AMD9950X3D-002 - Machine Profile

Collect OS, CPU flags, topology, scheduler/core placement context, memory, governor/power mode, thermal state, and optional OpenVINO CPU visibility.

### AMD9950X3D-003 - Scalar, AVX2, and AVX-512 Dispatch Proof

Prove scalar, AVX2, and AVX-512 paths can be forced independently and receipts record selected CPU kernel path.

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

## Do Not

- Do not treat AVX2 proof as AVX-512 proof.
- Do not report short boost as sustained performance.
- Do not ignore cache-domain or scheduler context for X3D behavior.
- Do not treat CPU proof as GPU/NPU proof.
- Do not make performance claims without sustained-power receipts.
