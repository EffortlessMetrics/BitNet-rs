# Apple M4 Mac mini Validation Profile

## Purpose

This file defines the machine profile and probe bundle for the Apple M4 Mac mini validation lane. It is a Metal-first profile with MPSGraph as a graph/reference lane and CPU/NEON as fallback/parity.

M4-002 is docs and artifact prep only. It records stable machine facts and planned probe artifact paths before any Metal kernels, MPSGraph graph execution, CPU/Metal parity, receipts, or benchmarks.

Roadmap:

```text
docs/specs/apple-m4-mac-mini-roadmap.md
```

## Hardware Baseline

Known public configuration classes:

| Machine class | Chip | CPU | GPU | Unified memory | Memory bandwidth class |
|---|---|---:|---:|---|---:|
| Base M4 Mac mini | Apple M4 | 10-core CPU | 10-core GPU | 16GB, configurable to 24GB or 32GB | 120 GB/s |
| M4 Pro Mac mini | Apple M4 Pro | 12-core or configurable 14-core CPU | 16-core or configurable 20-core GPU | up to 64GB | 273 GB/s |

The probe bundle must record the actual machine configuration. Do not infer base M4 values for an M4 Pro machine, and do not infer GPU core count or memory size from the lane name.

Required recorded facts:

- macOS version and kernel/build.
- Native macOS vs virtualized execution.
- Chip name: Apple M4 or Apple M4 Pro.
- CPU core count.
- GPU core count when visible from system tools or confirmed machine spec.
- Unified memory size.
- Memory bandwidth class when known from confirmed Apple machine docs/specs.
- Metal device visibility.
- MPSGraph lane availability notes.
- CPU/NEON visibility.
- Rust toolchain versions.

## Claim Boundary

| Lane | Meaning | Must not claim |
|---|---|---|
| `apple-m4-cpu-neon` | ARM64 CPU/NEON fallback and parity | Metal acceleration |
| `apple-m4-metal` | Native Metal compute path | MPSGraph or Neural Engine proof |
| `apple-m4-mpsgraph` | Graph/reference lane | Native Metal packed-kernel proof |
| Neural Engine | Only if resolved and receipt-backed | Never infer from MPSGraph alone |

Additional boundaries:

- Metal visibility is not Metal execution.
- Metal smoke is not CPU/Metal parity.
- CPU fallback cannot count as Metal execution.
- MPSGraph smoke cannot count as native handwritten Metal kernel proof.
- MPSGraph smoke cannot count as Neural Engine execution unless the resolved target is receipt-backed.
- Apple CPU/NEON is not AVX2 or AVX-512.
- Hardware probe artifacts are not BitNet proof artifacts.

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

echo "=== CPU / NEON ==="
sysctl machdep.cpu.brand_string || true
sysctl hw.physicalcpu hw.logicalcpu || true
sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu || true
sysctl hw.optional.neon || true

echo "=== Virtualization ==="
sysctl kern.hv_vmm_present || true

echo "=== Rust toolchain ==="
rustc --version
cargo --version
```

Record command output into the machine profile bundle without committing bulky machine-specific output in docs-only PRs.

## M4-004 Device Probe

M4-004 adds the dependency-free `bitnet-device-probe` Metal visibility probe:

```bash
cargo test --locked -p bitnet-device-probe --no-default-features --features metal
```

The probe records:

- `requested_backend = "apple-m4-metal"`.
- `selected_backend = "apple-m4-metal"` only when macOS reports Metal visibility on an Apple M4-family chip.
- `runtime_api = "metal"`.
- macOS version/build and kernel string.
- Apple chip name when reported by `system_profiler`.
- CPU and GPU core counts when visible from system tools.
- unified memory size from `sysctl hw.memsize`.
- native macOS vs virtualized macOS when `kern.hv_vmm_present` is available.
- Metal device name and support/family strings when visible.
- `fallback_used = false`.
- `proof_stage = "runtime_detected"` or `proof_stage = "runtime_unavailable"`.

This probe must not compile or dispatch Metal kernels, run MPSGraph, claim Neural Engine use, or claim BitNet inference.

## Expected Artifact Paths

Use the shared hardware artifact convention:

```text
ci/hardware/apple-m4-mac-mini/<date>/metal-probe.json
ci/hardware/apple-m4-mac-mini/<date>/cpu-neon-probe.json
ci/hardware/apple-m4-mac-mini/<date>/mpsgraph-probe.json
```

Use ISO dates such as `2026-05-05`. These are planned probe artifact paths only; do not add large artifacts for M4-002.

## Probe Receipt Shape

Metal probe artifacts record visibility only:

```json
{
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "resolved_device": {
    "chip": "Apple M4",
    "cpu_cores": 10,
    "gpu_cores": 10,
    "unified_memory": true,
    "unified_memory_bytes": 17179869184,
    "memory_bandwidth_gbps": 120,
    "native_or_virtualized": "native-macos",
    "metal_visible": true
  },
  "fallback_used": false,
  "proof_stage": "runtime_detected",
  "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-05/metal-probe.json"
}
```

CPU/NEON probe artifacts remain separate from Metal:

```json
{
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "resolved_device": {
    "chip": "Apple M4",
    "cpu_cores": 10,
    "neon_visible": true,
    "unified_memory": true,
    "native_or_virtualized": "native-macos"
  },
  "fallback_used": false,
  "proof_stage": "runtime_detected",
  "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-05/cpu-neon-probe.json"
}
```

MPSGraph probe artifacts remain graph/reference lane evidence only:

```json
{
  "requested_backend": "apple-m4-mpsgraph",
  "selected_backend": "apple-m4-mpsgraph",
  "runtime_api": "mpsgraph",
  "resolved_device": {
    "chip": "Apple M4",
    "mpsgraph_visible": true,
    "resolved_target": "unknown"
  },
  "fallback_used": false,
  "proof_stage": "runtime_detected",
  "artifact_path": "ci/hardware/apple-m4-mac-mini/2026-05-05/mpsgraph-probe.json"
}
```

The concrete values above are example base-M4 values. Generated artifacts must replace them with recorded machine facts, including M4 Pro values when applicable.

When a later artifact claims BitNet progress, it must also include the BitNet contract fields:

```json
{
  "model": {
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "tokenizer": "llama3"
  },
  "bitnet": {
    "kernel_family": "i2_s|tl1|qk256|openvino_graph",
    "execution_phase": "probe|smoke|parity|prefill|decode"
  }
}
```

M4-002 probe artifacts do not claim BitNet inference, Metal kernel execution, MPSGraph graph execution, or Neural Engine execution.

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
