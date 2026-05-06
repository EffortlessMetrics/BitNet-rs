# Intel Core i5-8250U CPU Roadmap

## Purpose

This lane validates the CPU-first BitNet path on low-power AVX2 mobile hardware.

Primary proof label:

```text
intel-i5-8250u-cpu-avx2
```

Secondary/deferred reference label:

```text
intel-uhd-620-openvino-gpu
```

The i5-8250U machine is not a GPU/NPU validation box. It is the conservative CPU path control box for scalar correctness, AVX2 dispatch, strict CPU proof runs, receipts, and sustained-load behavior.

## Hardware Baseline

Expected i5-8250U facts:

| Property | Expected value |
|---|---|
| Processor | Intel Core i5-8250U |
| Generation | 8th Gen mobile |
| Former codename | Kaby Lake R |
| Launch | Q3 2017 |
| Cores / threads | 4 / 8 |
| Base / turbo | 1.60 GHz / 3.40 GHz |
| Cache | 6 MB Smart Cache |
| TDP | 15 W |
| cTDP-up / cTDP-down | 25 W / 10 W |
| Lithography | 14 nm |
| Max memory | 32 GB |
| Memory types | DDR4-2400 or LPDDR3-2133, dual channel |
| Max memory bandwidth | 37.5 GB/s |
| CPU ISA | SSE4.1, SSE4.2, AVX2 |
| AVX-512 | Not supported |
| Integrated GPU | Intel UHD Graphics 620 |
| iGPU device ID | 0x5917 |

The 8250U should be treated as an AVX2-only CPU lane. Do not assume AVX-512, BF16, AMX, or modern desktop CPU behavior.

## Claim Boundary

- CPU detection is not CPU kernel proof.
- AVX2 detection is not AVX2 kernel proof.
- CPU proof is not GPU proof.
- UHD 620 OpenVINO or OpenCL visibility does not count as CPU-path progress.
- Short turbo performance is not sustained performance.
- CPU receipts must record scalar versus AVX2 selection.
- GPU/NPU fallback cannot be involved in strict CPU proof.

## Validation Levels

| Level | Evidence | Allowed claim |
|---|---|---|
| 0 | CPU model detected | i5-8250U detected |
| 1 | Runtime feature probe records AVX2 and no AVX-512 | CPU feature profile recorded |
| 2 | Scalar and AVX2 kernel smoke pass | CPU kernel smoke tested |
| 3 | Strict CPU inference receipt validates | CPU proof receipt backed |
| 4 | Sustained-load baseline exists | Low-power mobile baseline recorded |

## What This Box Is Good For

- CPU scalar parity.
- CPU AVX2 parity.
- Strict loader proof runs.
- Small deterministic prompts.
- Receipt schema validation.
- Thermal-throttle-aware long-run baselines.
- Low-memory and older-laptop operability checks.

## What This Box Should Not Own

- GPU performance claims.
- NPU work.
- OpenVINO NPU validation.
- AVX-512 work.
- BF16 or AMX CPU assumptions.
- Large parallel CI jobs.
- Large-model throughput claims without thermal context.

## Optional UHD 620 Reference Lane

The integrated Intel UHD Graphics 620 can be probed as an optional OpenVINO/OpenCL reference device.

Expected label:

```text
intel-uhd-620-openvino-gpu
```

Rules:

- UHD 620 work is deferred and must not block CPU proof.
- OpenVINO GPU visibility is not BitNet GPU acceleration.
- On an iGPU-only machine, OpenVINO may expose UHD 620 as `GPU.0`.
- Any UHD 620 claim requires selected-device receipts.

## Receipt Fields

Minimum CPU proof receipt:

```json
{
  "requested_backend": "cpu",
  "selected_backend": "intel-i5-8250u-cpu-avx2",
  "fallback_backend": null,
  "fallback_used": false,
  "cpu": {
    "model": "Intel Core i5-8250U",
    "cores": 4,
    "threads": 8,
    "avx2": true,
    "avx512": false,
    "selected_kernel_path": "avx2"
  },
  "power_thermal": {
    "mode": "...",
    "frequency_mhz": null,
    "temperature_c": null,
    "phase": "cold_turbo|warm_sustained|throttled"
  }
}
```

## Work Plan

### KBL8250U-001 - Add CPU Lane Docs

Docs/tracking only. Add backend status, roadmap, and hardware validation profile.

### KBL8250U-002 - Machine Profile

Collect OS, CPU flags, core/thread count, memory, governor/power mode, thermal state, OpenVINO CPU visibility, and optional UHD 620 visibility.

### KBL8250U-003 - Scalar and AVX2 Dispatch Proof

Prove AVX2 is available, AVX-512 is unavailable, scalar and AVX2 paths can be forced independently, and receipts record selected CPU kernel path.

### KBL8250U-004 - Strict CPU Proof Run

Run strict CPU inference proof with no minimal GGUF fallback, no GPU fallback, and no mock path.

### KBL8250U-005 - Sustained-Load Baseline

Separate cold turbo, warm sustained, and throttled performance, with frequency, temperature, power/governor mode, and duration when available.

## Relationship To Other Hardware Lanes

| Machine | Role |
|---|---|
| i5-8250U | CPU truth and sustained AVX2 mobile baseline |
| Arc A770 | Intel discrete GPU kernel path |
| Core Ultra 7 258V | Lunar Lake CPU/iGPU/NPU split |

The 8250U answers:

```text
Can the CPU-first path work honestly on older, low-power AVX2 hardware?
```
