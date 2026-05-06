# AMD Ryzen 7 5700X CPU Roadmap

## Purpose

This lane validates the mainstream AM4 / Zen 3 / AVX2 desktop CPU path for BitNet-rs.

Primary proof label:

```text
amd-5700x-cpu-avx2
```

Secondary comparison label:

```text
amd-5700x-cpu-scalar
```

The 5700X lane is CPU-only. It is not a GPU or NPU acceleration lane.

## Hardware Baseline

| Property | Expected value |
|---|---|
| CPU | AMD Ryzen 7 5700X |
| Architecture | Zen 3 / Vermeer |
| Socket | AM4 |
| Cores / threads | 8 / 16 |
| Base / boost | 3.4 GHz / up to 4.6 GHz |
| L3 cache | 32 MB |
| TDP | 65 W |
| Memory | DDR4-3200 |
| PCIe | PCIe 4.0 |
| Extensions | AVX2, AVX, FMA3 |
| Integrated graphics | None |

Do not treat this as an AVX-512 path.

## Claim Boundary

- AVX2 detection is not AVX2 kernel proof.
- 5700X has no AVX-512 lane.
- CPU proof is not GPU/NPU proof.
- DDR4/AM4 results are not directly comparable to DDR5/AM5 without memory context.
- Performance claims require sustained benchmark receipts.
- GPU/NPU fallback cannot be involved in strict CPU proof.

## Validation Levels

| Level | Evidence | Allowed claim |
|---|---|---|
| 0 | CPU model detected | 5700X detected |
| 1 | Runtime feature probe records AVX2 and no AVX-512 | CPU feature profile recorded |
| 2 | Scalar and AVX2 kernel smoke pass | CPU kernel smoke tested |
| 3 | Strict CPU inference receipt validates | CPU proof receipt backed |
| 4 | DDR4/AM4 sustained baseline exists | Mainstream desktop CPU benchmark recorded |

## Receipt Fields

Minimum CPU proof receipt:

```json
{
  "machine_id": "amd-5700x",
  "requested_backend": "cpu",
  "selected_backend": "amd-5700x-cpu-avx2",
  "fallback_backend": null,
  "fallback_used": false,
  "cpu": {
    "vendor": "AMD",
    "model": "Ryzen 7 5700X",
    "architecture": "Zen 3",
    "cores": 8,
    "threads": 16,
    "l3_cache_bytes": 33554432,
    "avx2_detected": true,
    "avx512_detected": false,
    "tdp_watts": 65
  }
}
```

## Work Plan

### AMD5700X-001 - Add CPU Lane Docs

Docs/tracking only. Add backend status, roadmap, and hardware validation profile.

### AMD5700X-002 - Machine Profile

Collect OS, CPU flags, topology, memory, governor/power mode, thermal state, AM4/DDR4 context, and optional OpenVINO CPU visibility.

### AMD5700X-003 - Scalar and AVX2 Dispatch Proof

Prove scalar and AVX2 paths can be forced independently and receipts record selected CPU kernel path.

### AMD5700X-004 - Strict CPU Proof Run

Run strict CPU proof with no GPU/NPU fallback, no mock path, and no hidden loader fallback.

### AMD5700X-005 - DDR4/AM4 Sustained Benchmark Baseline

Record DDR4/AM4 memory context, sustained frequency, temperature if available, power mode, and duration.

## Relationship To Other CPU Lanes

| Machine | Role |
|---|---|
| i5-8250U | Low-power Intel AVX2 mobile baseline |
| Ryzen 7 5700X | Mainstream AMD AVX2 desktop baseline |
| Ryzen 9 9950X3D | Modern AMD AVX-512 and large-cache desktop baseline |
| M4 Mac mini | ARM64/NEON and Metal ecosystem comparison |

The 5700X answers:

```text
How does the CPU-first path behave on mainstream AM4 / DDR4 / Zen 3 AVX2 hardware?
```

## Do Not

- Do not assume AVX-512.
- Do not treat CPU proof as GPU/NPU proof.
- Do not compare directly to DDR5/AM5 or X3D results without memory/cache context.
- Do not make performance claims without sustained benchmark receipts.
