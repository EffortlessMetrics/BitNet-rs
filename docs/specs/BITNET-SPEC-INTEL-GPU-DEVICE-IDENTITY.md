# BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines identity fields; no promotion.
Policy impact: No exception.

## Purpose

Normalize identity evidence across A770, Arc 140V, OpenCL, Level Zero,
OpenVINO GPU, and system telemetry so proof receipts cannot inherit claims from
another Intel device or runtime.

## Required identity fields

Every Intel GPU proof receipt must record, or explicitly mark unavailable:

- OS and kernel/build;
- native, WSL, container, or virtualized context;
- GPU name;
- GPU family;
- PCI ID;
- driver version;
- OpenCL platform and device index;
- Level Zero adapter identity;
- OpenVINO available devices;
- OpenVINO `GPU.X` full device name;
- VRAM or shared memory;
- ReBAR for A770;
- PCIe link width and generation for A770;
- Linux render-node and permission context;
- power, thermal, and utilization tool availability.

## Device-specific expectations

| Device family | Required selected identity |
| --- | --- |
| A770 | Full Arc A770 device name, PCI ID `0x56A0` when exposed, discrete VRAM, OpenCL platform/device, and PCIe/ReBAR facts where available. |
| Arc 140V | Full Arc 140V or Core Ultra 7 258V integrated GPU identity, PCI ID `0x64A0` when exposed, shared-memory context, and OpenVINO `GPU.X` mapping when OpenVINO is used. |
| OpenVINO GPU | `GPU.X` selector, OpenVINO runtime or GenAI API version, available-device list, and resolved full device name. |
| Level Zero | Adapter name, driver/runtime identity, and explicit candidate status unless promoted by a later spec. |

## Unknown fields

Unknown telemetry is acceptable only when the receipt records the attempted
source and reason it is unavailable. Missing telemetry cannot silently become a
performance, power, residency, or selected-device claim.
