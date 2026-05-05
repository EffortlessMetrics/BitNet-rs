# Apple M4 Mac mini Validation Profile

## Purpose

This file defines the hardware data bundle for the Apple M4 Mac mini validation lane. It is a Metal-first profile with MPSGraph as a reference lane and CPU/NEON as fallback/parity.

Roadmap:

```text
docs/specs/apple-m4-mac-mini-roadmap.md
```

## Hardware Baseline

Base M4 Mac mini:

- Chip: Apple M4.
- CPU: 10-core CPU.
- GPU: 10-core GPU.
- Neural Engine: 16-core Neural Engine.
- Unified memory: 16GB configurable to 24GB or 32GB.
- Memory bandwidth: 120 GB/s.
- GPU features: Dynamic Caching, hardware ray tracing, mesh shading.

M4 Pro Mac mini:

- CPU: 12-core or configurable 14-core CPU.
- GPU: 16-core or configurable 20-core GPU.
- Neural Engine: 16-core Neural Engine.
- Unified memory: up to 64GB.
- Memory bandwidth: 273 GB/s.

Record the actual machine configuration in every receipt.

## Claim Boundary

- Metal visibility is not Metal execution.
- Metal smoke is not CPU/Metal parity.
- CPU fallback cannot count as Metal execution.
- MPSGraph smoke cannot count as native handwritten Metal kernel proof.
- MPSGraph smoke cannot count as Neural Engine execution unless the resolved target is receipt-backed.
- Apple CPU/NEON is not AVX2 or AVX-512.

## macOS Probe Bundle

```bash
set -eux

echo "=== macOS ==="
sw_vers
uname -a

echo "=== Hardware ==="
system_profiler SPHardwareDataType

echo "=== Graphics / Metal ==="
system_profiler SPDisplaysDataType
system_profiler SPMetalDataType || true

echo "=== Memory ==="
vm_stat
sysctl hw.memsize
sysctl machdep.cpu.brand_string || true
sysctl hw.optional.neon || true

echo "=== Rust / toolchain ==="
rustc --version
cargo --version
```

## First Metal Receipt

The first useful receipt is a Metal smoke proof:

```json
{
  "hardware": "apple-m4-mac-mini",
  "requested_backend": "apple-m4",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "chip": "Apple M4",
  "gpu_cores": 10,
  "unified_memory": true,
  "memory_bandwidth_gbps": 120,
  "fallback_used": false,
  "status": "kernel_smoke_tested"
}
```

For M4 Pro, this must record the actual core and memory configuration.

## Optional MPSGraph Receipt

```json
{
  "hardware": "apple-m4-mac-mini",
  "requested_backend": "apple-m4-mpsgraph",
  "selected_backend": "apple-m4-mpsgraph",
  "runtime_api": "mpsgraph",
  "resolved_target": "unknown",
  "fallback_used": false,
  "status": "kernel_smoke_tested"
}
```

This is graph/reference proof only, not native Metal kernel proof.

## Benchmark Notes

Benchmarks must record:

- macOS version.
- Chip and exact M4/M4 Pro configuration.
- Unified memory size.
- Selected backend.
- Fallback status.
- Cold and warm timing.
- Thermal/power context when available.

Do not compare M4 unified-memory results directly to discrete VRAM GPUs without memory context.
