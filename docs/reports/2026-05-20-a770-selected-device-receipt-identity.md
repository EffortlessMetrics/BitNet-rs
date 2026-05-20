# A770 Selected-Device Receipt Identity

Status: diagnostic
Owner: intel-a770 campaign
Created: 2026-05-20
Linked proposal: n/a
Linked specs: `docs/specs/intel-arc-a770-gpu-roadmap.md`, `docs/specs/a770-bitnet-claim-boundary.md`
Linked ADRs: n/a
Linked plan: `docs/tracking/campaigns/intel-a770/active.toml`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No public support-tier promotion.
Policy impact: No policy exception.

## Purpose

This report records selected-device receipt identity for the committed A770
native OpenCL smoke and parity path before any benchmark, trusted partial
acceleration, full residency, or BitNet inference claim.

## Receipt Index

| Work item | Receipt | Proof family | Kernel | Stage |
| --- | --- | --- | --- | --- |
| A770-005 | `ci/hardware/amd-5700x-intel-a770/2026-05-20/a770-opencl-tiny-smoke.json` | `a770_opencl_tiny_vector_add_smoke` | `tiny_vector_add` | `kernel_smoke_tested` |
| A770-006 | `ci/hardware/amd-5700x-intel-a770/2026-05-20/a770-opencl-matmul-i2s-parity.json` | `a770_opencl_matmul_i2s_cpu_parity` | `matmul_i2s` | `cpu_opencl_parity_tested` |
| A770-007 | `ci/hardware/amd-5700x-intel-a770/2026-05-20/a770-selected-device-receipt-identity.json` | `a770_selected_device_receipt_identity` | n/a | `selected_device_receipt_identity_recorded` |

## Selected Device

The committed smoke and parity receipts agree on the selected route identity:

```text
requested_backend = intel-arc-a770
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
platform_index = 0
device_index = 0
platform_name = Intel(R) OpenCL Graphics
vendor = Intel(R) Corporation
driver_version = 32.0.101.8801
fallback_used = false
```

## What This Proves

This proves only that selected-device A770 OpenCL receipt identity exists for
the committed diagnostic smoke/parity path:

- A770-005 executed a tiny vector-add OpenCL smoke on the selected A770 route.
- A770-006 executed the existing `bitnet-kernels` `matmul_i2s` OpenCL source on
  the selected A770 route and matched the CPU reference with `max_abs_error=0`.
- Both receipts preserve `fallback_used=false`.

## Claim Boundary

The A770-007 receipt identity report does not promote:

- A770 trusted partial acceleration;
- BitNet inference on A770;
- official BitNet QK256 production semantics;
- selected attention residency;
- resident KV decode;
- attention score, softmax, or value-mix residency;
- full A770 residency;
- A770 performance speedup;
- completion.

## Next Gate

A770-008 may create a benchmark baseline for the validated diagnostic
kernel/subgraph. It must remain diagnostic until quality, model, route,
fallback, resource, and history gates are satisfied by later receipts.
