# BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY

## Purpose

Normalize device identity for A770, Arc 140V, OpenCL, Level Zero, OpenVINO GPU,
and system telemetry so receipts can prove the selected Intel route.

## Required identity fields

Receipts must record, when available:

- OS and kernel/build.
- Native, WSL, container, or virtualized context.
- GPU name and GPU family.
- PCI ID.
- Driver version.
- OpenCL platform/device index.
- Level Zero adapter identity.
- OpenVINO available devices.
- OpenVINO `GPU.X` full device name.
- VRAM or shared-memory capacity.
- ReBAR state for A770.
- PCIe link width/generation for A770.
- Linux render-node and permission context.
- Power, thermal, and utilization tool availability.

## Identity boundaries

A device name match alone is not enough for a promoted claim when PCI/runtime
identity is available. Missing telemetry is allowed only when receipts explicitly
record that the tool or permission was unavailable.
