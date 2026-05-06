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

## M4-010 CPU/NEON BitNet Reference

The Apple CPU/NEON BitNet reference proof uses the CLI `--json-out` artifact as
the receipt path:

```bash
BITNET_DISABLE_MINIMAL_LOADER=1 \
BITNET_STRICT_MODE=1 \
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- run \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --prompt "Answer with a single digit: 2+2=" \
  --max-tokens 1 \
  --temperature 0.0 \
  --greedy \
  --device apple-m4-cpu-neon \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/strict-bitnet-cpu-neon-proof.json
```

The receipt must record `requested_backend`, `selected_backend`,
`runtime_api`, `fallback_used`, model repo/file/SHA-256, tokenizer source,
loader mode, kernel family, execution phase, CPU features, and generated token
count. The M4-010 reference path records the canonical GGUF packed I2_S layout
and selected scalar reference kernel explicitly; NEON may be present as a CPU
feature, but this proof must not claim a NEON-optimized packed kernel unless one
is actually selected. It also does not claim Metal BitNet execution, QK256
optimization on Apple Silicon, or Apple CPU performance.

## M4-011 Native Metal I2_S Smoke/Parity

The first BitNet-specific native Metal proof is a tiny I2_S-adjacent parity
fixture, not full model inference. It uses a 1x4 output fixture with `k=32`,
canonical I2_S packed bytes, and a `u32_le_words_from_i2s_bytes` transport buffer
for Metal storage-buffer alignment. The receipt must say both things: the source
layout is packed I2_S, and the Metal test transports those bytes as little-endian
`u32` words.

Run the non-live contract tests with:

```bash
cargo test --locked -p bitnet-kernels \
  --no-default-features --features metal \
  --test metal_tiny_smoke i2s -- --nocapture
```

Run the live M4 receipt-backed proof with:

```bash
BITNET_RUN_M4_METAL_I2S_PARITY=1 \
BITNET_M4_METAL_I2S_PARITY_RECEIPT=ci/hardware/apple-m4-mac-mini/<date>/metal-i2s-parity.json \
cargo test --locked -p bitnet-kernels \
  --no-default-features --features metal \
  --test metal_tiny_smoke tiny_m4_metal_i2s_matches_cpu_neon_reference_when_enabled -- --nocapture
```

The receipt may claim only that `tiny_metal_i2s_parity` runs on
`apple-m4-metal`, consumes the declared packed I2_S fixture layout without CPU
fallback, and matches the Apple CPU/NEON reference lane for that fixture. It must
not claim full BitNet Metal inference, QK256 on Metal, benchmark performance,
server inference, MPSGraph execution, or Neural Engine execution.

## M4-012 TL1 Apple CPU/NEON Investigation

TL1 is currently an Apple CPU/NEON-oriented BitNet layout investigation item,
not a native Metal proof. The TL1 quantizer records unsigned 2-bit LUT codes
packed four values per byte, per-block scales, and optional zero points for
asymmetric quantization. The default Apple contract uses:

```json
{
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "bitnet": {
    "kernel_family": "tl1",
    "execution_phase": "investigation",
    "layout_source": "tl1_reference",
    "fallback_layout": null
  },
  "layout": {
    "source": "tl1_reference",
    "transport_layout": "tl1_packed_u2_codes_with_scales",
    "conversion_boundary": "tl1_to_metal_transport_not_proven",
    "consumes_packed_tl1_directly_on_metal": false,
    "dequantizes_before_compute": true
  },
  "fallback_used": false
}
```

The contract may claim only that TL1 CPU/NEON behavior and the current Metal
conversion boundary are documented or receipt-backed. It must not claim TL1
runs natively on Metal until a later receipt proves that Metal consumes the TL1
layout directly or names the exact conversion/dequantization path before
compute.

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
