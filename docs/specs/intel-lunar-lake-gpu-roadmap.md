# Intel Lunar Lake Arc 140V GPU Roadmap

## Purpose

This document defines the Intel Arc 140V integrated GPU lane for the Core Ultra 7 258V validation laptop. It is separate from the Lunar Lake NPU lane.

Primary target:

```text
intel-arc-140v-opencl
```

Reference target:

```text
intel-arc-140v-openvino-gpu
```

The Arc 140V lane exists to compare CPU AVX2, shared-memory iGPU OpenCL, OpenVINO GPU, and NPU/OpenVINO behavior on the same Lunar Lake machine.

## Hardware Profile

Expected 258V integrated GPU facts:

| Property | Expected value |
|---|---|
| GPU | Intel Arc 140V |
| Architecture | Xe2-LPG |
| Xe-cores | 8 |
| Vector engines per Xe-core | 8 |
| INT8 peak | 64 TOPS |
| PCI device ID | 0x64A0 |
| Memory kind | Shared LPDDR5X system memory |
| Runtime support | OpenCL 3.0, Level Zero, OpenVINO GPU, DirectML, ONNX RT, WebGPU/WebNN |

Arc 140V is not a small A770. It is an integrated GPU with shared memory and laptop power/thermal constraints. Performance claims must record memory pressure and power mode.

## Claim Boundary

Do not claim Arc 140V execution from device visibility alone.

| Evidence | Allowed claim |
|---|---|
| OpenCL or Level Zero sees Arc 140V | Runtime detected |
| OpenVINO sees `GPU.0` with Arc 140V full name | Reference runtime detected |
| Tiny OpenCL kernel executes | Kernel smoke tested |
| CPU/iGPU parity passes | Parity tested |
| Receipt records selected Arc 140V backend and no fallback | Receipt backed |
| Benchmark artifact includes power/memory context | Performance claim allowed |

OpenVINO GPU graph smoke is not native OpenCL BitNet kernel proof.

CPU fallback cannot count as Arc 140V execution.

## Backend Labels

Native OpenCL:

```text
requested_backend = "intel-arc-140v"
selected_backend = "intel-arc-140v-opencl"
runtime_api = "opencl"
pci_device_id = "0x64A0"
memory_kind = "shared-system-memory"
```

OpenVINO GPU reference:

```text
requested_backend = "intel-arc-140v"
selected_backend = "intel-arc-140v-openvino-gpu"
openvino_device = "GPU.0"
```

Do not use plain `intel`, `gpu`, or `oneapi` as proof labels.

## Runtime Paths

### Native OpenCL Path

Milestones:

1. OpenCL and Level Zero visibility on the 258V laptop.
2. Strict Arc 140V selected-device identity.
3. Tiny OpenCL kernel smoke.
4. CPU/iGPU parity for one minimal kernel or subgraph.
5. Shared-memory pressure and laptop power benchmark receipt.

### OpenVINO GPU Reference Path

On the 258V laptop, Arc 140V is expected to be OpenVINO `GPU.0` when visible.

Milestones:

1. OpenVINO `available_devices` includes `GPU`.
2. `GPU.0` resolves to Arc 140V full device name.
3. Tiny fixed-shape graph compiles to `GPU.0`.
4. Output compares against CPU expected output.
5. Receipt records OpenVINO version, GPU.0 identity, shared-memory context, and fallback status.

## Probe Shape

Suggested probe result:

```rust
pub struct IntelArc140vProbe {
    pub proof_stage: String,
    pub requested_backend: String,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub available: bool,
    pub pci_device_id: Option<String>,
    pub identity_evidence: Vec<String>,
    pub opencl_available: bool,
    pub opencl_platform_name: Option<String>,
    pub opencl_device_name: Option<String>,
    pub opencl_driver_version: Option<String>,
    pub level_zero_available: bool,
    pub openvino_gpu_visible: bool,
    pub openvino_gpu_device: Option<String>,
    pub openvino_gpu_full_name: Option<String>,
    pub shared_memory_bytes: Option<u64>,
    pub power_mode: Option<String>,
    pub fallback_used: bool,
    pub failure_reason: Option<String>,
}
```

For `ARC140V-002`, exact identity can come from OpenCL device name, Level Zero device name, Level Zero PCI/device ID `0x64A0`, or OpenVINO `GPU.0` full device name. Generic Intel GPU visibility is not sufficient.

## Receipt Fields

Minimum native OpenCL receipt:

```json
{
  "requested_backend": "intel-arc-140v",
  "selected_backend": "intel-arc-140v-opencl",
  "fallback_backend": null,
  "fallback_used": false,
  "runtime": {
    "api": "opencl",
    "device_name": "Intel(R) Arc(TM) 140V Graphics",
    "driver_version": "...",
    "pci_device_id": "0x64A0",
    "memory_kind": "shared-system-memory",
    "power_mode": "..."
  },
  "kernels_or_graphs": [
    "opencl_smoke"
  ]
}
```

Minimum OpenVINO GPU reference receipt:

```json
{
  "requested_backend": "intel-arc-140v-openvino-gpu",
  "selected_backend": "openvino-gpu",
  "openvino_device": "GPU.0",
  "full_device_name": "Intel(R) Arc(TM) 140V Graphics",
  "fallback_used": false,
  "graph": {
    "name": "tiny_matmul_f16",
    "shape_mode": "static"
  }
}
```

## Platform Relationship

The Core Ultra 7 258V laptop is a tri-device validation platform:

| Device | Proof lane |
|---|---|
| CPU | `cpu-avx2` correctness and fallback |
| Arc 140V integrated GPU | `intel-arc-140v-opencl` and `intel-arc-140v-openvino-gpu` |
| Intel AI Boost NPU | `intel-npu-openvino` |

The Arc 140V lane answers whether a BitNet shape should run on CPU AVX2, shared-memory iGPU OpenCL, or NPU/OpenVINO on the same laptop.

## Validation Bundle

The platform bundle lives in `docs/hardware/intel-258v-validation.md`.

It must collect:

- Native Windows, native Linux, or WSL context.
- CPU model, core count, AVX2 support, and thread count.
- 32GB LPDDR5X memory details when available.
- Arc 140V OpenCL and Level Zero visibility.
- OpenVINO `GPU.0` full device name.
- NPU OpenVINO visibility and NPU driver version.
- Power mode and thermal profile.
- Shared-memory pressure context.

## Work Plan

### ARC140V-001 - Add Integrated GPU Lane

Docs/tracking only. Add Arc 140V as a separate integrated GPU lane from the NPU lane.

### ARC140V-002 - Runtime Probe

Detect Arc 140V by device name or PCI ID 0x64A0, and record OpenCL, Level Zero, and OpenVINO GPU.0 visibility.

### ARC140V-003 - OpenVINO GPU Smoke

Compile and run a tiny fixed-shape graph on OpenVINO `GPU.0`.

Command shape:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- intel-arc-140v-openvino-gpu-smoke \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-openvino-gpu-smoke.json
```

The receipt must keep `requested_backend=intel-arc-140v`,
`selected_backend=intel-arc-140v-openvino-gpu`, `runtime_api=openvino`,
`runtime_device=GPU.0`, `fallback_used=false`, `bitnet_inference=false`, and
`qk256_decode=false`. It may claim tiny OpenVINO GPU graph smoke only when the
selected `GPU.0` full device name identifies Arc 140V and the graph output
matches the CPU expected output.

This is not native OpenCL execution. Native OpenCL starts at `ARC140V-004`.

### ARC140V-004 - OpenCL Kernel Smoke

Compile and run a tiny OpenCL kernel on Arc 140V.

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli,opencl \
  -- intel-arc-140v-opencl-smoke \
  --json-out ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-opencl-smoke.json
```

The receipt must keep `requested_backend=intel-arc-140v`,
`selected_backend=intel-arc-140v-opencl`, `runtime_api=opencl`,
`kernel_execution=true`, `fallback_used=false`, `bitnet_inference=false`, and
`qk256_decode=false` only when the selected OpenCL device resolves to Arc 140V
and the tiny vector-add output matches the CPU expected output.

### ARC140V-005 - CPU/iGPU Parity

Run one isolated kernel or subgraph through OpenCL and compare against CPU.

Use the post-mechanics 258V CPU reference bundle as the comparison anchor:

```text
ci/hardware/intel-258v/2026-05-08/cpu-reference-bundle-post-mechanics.json
```

The first live receipt is:

```text
ci/hardware/intel-258v/2026-05-08/arc-140v-opencl-parity.json
```

It runs the same tiny native OpenCL vector-add kernel as `ARC140V-004`,
but records the post-mechanics 258V CPU reference bundle as the comparison
anchor and promotes the proof stage to `parity_tested` only when the Arc 140V
OpenCL output matches the CPU reference within tolerance.

The first `ARC140V-005` receipt should stay narrow:

```text
requested_backend=intel-arc-140v
selected_backend=intel-arc-140v-opencl
runtime_api=opencl
proof_stage=parity_tested
kernel_execution=true
graph_execution=false
fallback_used=false
bitnet_inference=false
qk256_decode=false
```

It may claim only that the selected native OpenCL kernel or static subgraph
matches the selected 258V CPU reference within the declared tolerance. It must
not claim Arc 140V BitNet inference, packed QK256 decode, acceleration, or CPU
fallback as Arc proof.

### ARC140V-006 - Shared-Memory Benchmark Receipt

Benchmark a validated kernel/subgraph with memory pressure, power mode, runtime versions, and fallback status.

## Practical Direction

Use Arc 140V for:

```text
shared-memory iGPU OpenCL smoke
OpenVINO GPU.0 smoke
CPU vs iGPU comparison on the same 258V thermals
small static subgraph parity
```

Do not judge Arc 140V by A770 expectations. It is memory-shared and power-limited.

## Related Roadmaps

- `docs/specs/intel-lunar-lake-258v-platform-roadmap.md`
- `docs/specs/intel-lunar-lake-npu-roadmap.md`
- `docs/hardware/intel-258v-validation.md`

The Arc 140V lane owns OpenCL and OpenVINO `GPU.0` validation. The platform roadmap owns whole-laptop comparison, and the NPU roadmap owns OpenVINO `NPU` validation.
